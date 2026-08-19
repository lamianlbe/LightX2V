"""Per-step samplers for the LTX-2.x rectified-flow schedulers.

LTX-2 is a CONST / rectified-flow model in ComfyUI terms
(``x_t = (1 - sigma) * x0 + sigma * noise``), so ComfyUI routes its
``euler_ancestral`` sampler to the RF variant. The formulas below are the
per-step math of ComfyUI's k-diffusion samplers for that model class:

===========================  ===============================================
``euler``                    deterministic velocity step (LTX-2 default)
``euler_ancestral``          ``sample_euler_ancestral_RF``
``euler_ancestral_cfg_pp``   ``sample_euler_ancestral_cfg_pp`` (CONST branch)
===========================  ===============================================

The two ancestral steppers below take the x0 prediction (``denoised``) and
return the next latent. They are plain tensor functions with no scheduler
state, so they can be unit tested against ComfyUI numerically.

``euler`` is deliberately absent: it stays inline in
``LTX2Scheduler.step_post`` so its float32/bfloat16 rounding sequence remains
bit-identical to the update this scheduler shipped before samplers became
configurable.
"""

from typing import Optional, Tuple

import torch

SAMPLER_NAMES = ("euler", "euler_ancestral", "euler_ancestral_cfg_pp")

#: Samplers that inject fresh noise every step and therefore need both an
#: ancestral noise generator and the conditioning repin (see
#: ``repin_conditioned_latent``).
STOCHASTIC_SAMPLERS = ("euler_ancestral", "euler_ancestral_cfg_pp")

#: Samplers that need the raw unconditional x0 prediction in addition to the
#: guided one. ComfyUI runs the negative pass for these even at cfg == 1
#: (``disable_cfg1_optimization=True``), so they are never equivalent to their
#: non-cfg_pp counterpart.
UNCOND_SAMPLERS = ("euler_ancestral_cfg_pp",)


def validate_sampler(name: str, *, field: str = "sampler") -> str:
    if name not in SAMPLER_NAMES:
        raise ValueError(f"{field} must be one of {list(SAMPLER_NAMES)}, got {name!r}")
    return name


def euler_ancestral_rf_step(
    x: torch.Tensor,
    denoised: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    noise: Optional[torch.Tensor],
    *,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """One step of ComfyUI ``sample_euler_ancestral_RF``.

    Steps down to ``sigma_down`` (an ``eta``-weighted interpolation toward
    ``sigma_next``), then renoises back up to ``sigma_next``.
    """
    sigma = sigma.to(torch.float32)
    sigma_next = sigma_next.to(torch.float32)
    if float(sigma_next) == 0.0:
        return denoised

    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down

    ratio = sigma_down / sigma
    x_next = ratio * x + (1.0 - ratio) * denoised

    if eta > 0:
        if noise is None:
            raise ValueError("euler_ancestral needs a noise tensor when eta > 0")
        renoise_coeff = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0) ** 0.5
        # Cast before scaling: a low-precision noise tensor times a Python float
        # stays low precision (scalars do not promote), which would round the
        # renoise term an extra time.
        x_next = (alpha_next / alpha_down) * x_next + noise.to(torch.float32) * s_noise * renoise_coeff
    return x_next


def get_ancestral_step(sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """ComfyUI k-diffusion ``get_ancestral_step``.

    Returns ``(sigma_down, sigma_up)``: the level to step down to, and how much
    fresh noise to add back.
    """
    if not eta:
        return sigma_to, torch.zeros_like(sigma_to)
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def euler_ancestral_cfg_pp_step(
    x: torch.Tensor,
    denoised: torch.Tensor,
    uncond_denoised: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    noise: Optional[torch.Tensor],
    *,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """One step of ComfyUI ``sample_euler_ancestral_cfg_pp`` for CONST models.

    CFG++ decouples the *position* term (``alpha_t * denoised``, i.e. the
    post-guidance conditional x0) from the *direction* term ``d``, which is
    built from the raw **unconditional** x0. For rectified-flow models the
    half-logSNR factors collapse to ``alpha = 1 - sigma``.
    """
    sigma = sigma.to(torch.float32)
    sigma_next = sigma_next.to(torch.float32)
    if float(sigma_next) == 0.0:
        return denoised
    if float(sigma) >= 1.0:
        raise ValueError(
            "euler_ancestral_cfg_pp is undefined at sigma >= 1.0 for rectified-flow models "
            "(alpha = 1 - sigma hits 0, which is where ComfyUI silently yields inf/NaN). "
            "Start the sigma schedule below 1.0, e.g. 0.99."
        )

    alpha_s = 1.0 - sigma
    alpha_t = 1.0 - sigma_next

    d = (x - alpha_s * uncond_denoised) / sigma
    sigma_down, sigma_up = get_ancestral_step(sigma / alpha_s, sigma_next / alpha_t, eta=eta)
    sigma_down = alpha_t * sigma_down
    x_next = alpha_t * denoised + sigma_down * d

    if eta > 0 and s_noise > 0:
        if noise is None:
            raise ValueError("euler_ancestral_cfg_pp needs a noise tensor when eta > 0 and s_noise > 0")
        x_next = x_next + alpha_t * noise.to(torch.float32) * s_noise * sigma_up
    return x_next


def repin_conditioned_latent(
    latent: torch.Tensor,
    *,
    clean_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    cond_noise: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Re-impose partially-pinned conditioning after ancestral noise injection.

    The ancestral samplers add fresh noise across the *whole* latent every
    step. Near sigma = 1 a single step can replace most of a conditioned
    frame's content, which destroys i2v / keyframe anchoring. ComfyUI avoids
    this in its ``KSamplerX0Inpaint`` wrapper, which re-blends the masked
    region toward the correctly-noised clean latent on every model call; this
    is the equivalent::

        ideal  = lerp(clean, cond_noise, mask * sigma_next)
        latent = latent * mask + ideal * (1 - mask)

    ``cond_noise`` must be a *fixed* per-run tensor (ComfyUI reuses the
    sampler's initial noise) so the pinned region stays on one consistent
    noise trajectory. A no-op where ``mask == 1``; the deterministic Euler
    sampler does not need it at all.
    """
    ideal = torch.lerp(clean_latent.float(), cond_noise.float(), denoise_mask * sigma_next.to(torch.float32))
    mask = denoise_mask.to(torch.float32)
    return (latent.float() * mask + ideal * (1.0 - mask)).to(latent.dtype)
