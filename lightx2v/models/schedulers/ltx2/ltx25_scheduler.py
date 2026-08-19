import torch

from lightx2v.models.schedulers.ltx2.scheduler import LTX2Scheduler


class LTX25Scheduler(LTX2Scheduler):
    """LTX-2.5 scheduler with distilled stage-1 ancestral Euler sampling.

    Stage 1 defaults to the released ancestral settings and stage 2 to the
    deterministic LTX-2 Euler update. These are now plain defaults: ``sampler``
    / ``sampler_stage1`` / ``sampler_stage2`` in the config override them, as do
    ``sampler_eta`` / ``sampler_s_noise``.

    ``DEFAULT_REPIN_MODE`` stays ``"clean"`` here so released LTX-2.5 numerics
    are unchanged. ``"noised"`` (the LTX-2.3 default) is the ComfyUI-equivalent
    behaviour and is the better choice for image/keyframe conditioning; set
    ``ancestral_repin_mode`` explicitly to switch.
    """

    DEFAULT_SAMPLER_STAGE1 = "euler_ancestral"
    DEFAULT_SAMPLER_STAGE2 = "euler"
    DEFAULT_REPIN_MODE = "clean"
    #: The released LTX-2.5 loop drew its ancestral noise in the latent dtype.
    #: Keep that, or bf16 runs would shift off the shipped numerics.
    ANCESTRAL_NOISE_IN_LATENT_DTYPE = True

    def _prepare_video_latents(self, *args, **kwargs) -> None:
        super()._prepare_video_latents(*args, **kwargs)

        # LTX-2.5 treats the target's first causal latent frame as a standalone
        # pixel-frame token class. The source pipeline marks it even when there
        # are no generated keyframe slots; ordinary image/reference tokens stay
        # unmarked.
        state = self.video_latent_state
        keyframes_mask = torch.zeros_like(state.denoise_mask)
        _, frames, _, _ = self.video_latent_shape_orig
        main_tokens = self._video_main_num_tokens
        if frames <= 0 or main_tokens is None or main_tokens % frames != 0:
            raise ValueError(f"Cannot determine LTX-2.5 first-frame token count from frames={frames}, main_tokens={main_tokens}")
        keyframes_mask[: main_tokens // frames] = 1.0
        state.keyframes_mask = keyframes_mask
