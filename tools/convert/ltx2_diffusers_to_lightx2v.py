#!/usr/bin/env python3
"""Convert a diffusers-layout LTX-2.x repo into the single-file layout LightX2V loads.

LightX2V's LTX-2 runner resolves the DiT, the video VAE, the audio VAE, the
vocoder and the caption/embedding projection to *one* checkpoint (all four
``*_checkpoint_key`` class attributes are ``None``, so they fall through to
``_component_checkpoint_path()``). The components are separated at load time by
key prefix, which is the ComfyUI / original-Lightricks convention:

    model.diffusion_model.*                     DiT
    model.diffusion_model.*_embeddings_connector.*   text embedding connectors
    text_embedding_projection.*_aggregate_embed.*    embedding feature extractor
    vae.{encoder,decoder,per_channel_statistics}.*   video VAE
    audio_vae.{encoder,decoder,per_channel_statistics}.*
    vocoder.*                                   vocoder (+ BWE, mel_stft)

A diffusers export stores each of those in its own subdirectory with bare keys,
so the conversion is a prefix remap plus a merge. Nothing is renamed inside a
component, and no tensor is transformed -- this is a byte-exact repack.

The spatial upscalers stay separate (LightX2V loads them via
``upsampler_original_ckpt``) and keep their keys verbatim.

Both the merged file and each upscaler file must carry a ``config`` entry in
their safetensors ``__metadata__``: the video VAE, audio VAE, vocoder and
upscaler loaders all read their architecture from there and have no fallback.
That metadata is assembled from the diffusers ``config.json`` files, which are
already shaped the way the configurators expect.

Memory: tensor payloads are copied as raw bytes straight from source to
destination in fixed-size chunks, so peak RSS stays flat regardless of model
size -- a 22B bf16 DiT converts in a few hundred MB of RAM.

Usage:
    python tools/convert/ltx2_diffusers_to_lightx2v.py \\
        --src  /path/to/10Eros_v1_Diffusers \\
        --out  /path/to/10Eros_v1_LightX2V \\
        --name 10eros-v1

    # then
    python -m lightx2v.infer --model_cls ltx2 --task t2av \\
        --model_path /path/to/10Eros_v1_LightX2V \\
        --config_json /path/to/10Eros_v1_LightX2V/lightx2v_ltx2_3.json ...

Verify an existing conversion without rewriting it:
    python tools/convert/ltx2_diffusers_to_lightx2v.py --out /path/to/out --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

CHUNK = 32 * 1024 * 1024  # 32 MiB streaming copy buffer

# --------------------------------------------------------------------------
# Component -> prefix mapping
#
# Each entry is (subdir, [(source_prefix, dest_prefix), ...]). A source prefix
# of "" means "prefix every key". Rules are tried longest-first so that e.g.
# ``audio_embeddings_connector.`` never falls through to a shorter rule.
# --------------------------------------------------------------------------

MERGED_COMPONENTS: List[Tuple[str, List[Tuple[str, str]]]] = [
    # The DiT: a pure prefix add. Verified key-for-key against LightX2V's
    # LTX2 weight modules -- there is no intra-component renaming.
    ("transformer", [("", "model.diffusion_model.")]),
    # Text embedding projection. Two things live here: the connectors (which
    # LightX2V reads out of the diffusion_model namespace) and the aggregate
    # embeds (which it reads out of text_embedding_projection.*). Note the
    # video connector is called ``embeddings_connector`` in diffusers and
    # ``video_embeddings_connector`` in the ComfyUI layout.
    (
        "text_embedding_projection",
        [
            ("audio_embeddings_connector.", "model.diffusion_model.audio_embeddings_connector."),
            ("embeddings_connector.", "model.diffusion_model.video_embeddings_connector."),
            ("video_aggregate_embed.", "text_embedding_projection.video_aggregate_embed."),
            ("audio_aggregate_embed.", "text_embedding_projection.audio_aggregate_embed."),
            ("aggregate_embed.", "text_embedding_projection.aggregate_embed."),
        ],
    ),
    # Video VAE. LightX2V's filter accepts bare ``encoder.``/``decoder.`` too,
    # but the audio VAE uses those same bare names, so both are namespaced to
    # keep the merged file unambiguous.
    ("vae", [("", "vae.")]),
    ("audio_vae", [("", "audio_vae.")]),
    # Vocoder: the filter strips exactly one ``vocoder.``, which is why the
    # inner ``vocoder.`` / ``bwe_generator.`` / ``mel_stft.`` groups all sit
    # one level down.
    ("vocoder", [("", "vocoder.")]),
]

# Subdirs whose config.json is merged into the main file's `config` metadata.
# Each of these is already nested under its own top-level key upstream.
CONFIG_COMPONENTS = ["vae", "audio_vae", "vocoder"]

# Upscalers: emitted as standalone files, keys untouched.
UPSCALERS = [
    ("spatial_upscaler", "spatial-upscaler-x2"),
    ("spatial_upscaler_x1_5", "spatial-upscaler-x1.5"),
]

# A spot-check of DiT keys that must survive the remap. Not exhaustive -- the
# full check is the count comparison in verify().
DIT_SENTINEL_KEYS = [
    "model.diffusion_model.patchify_proj.weight",
    "model.diffusion_model.audio_patchify_proj.weight",
    "model.diffusion_model.scale_shift_table",
    "model.diffusion_model.adaln_single.linear.weight",
    "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.transformer_blocks.0.audio_to_video_attn.to_k.weight",
    "model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_v.weight",
    "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.transformer_blocks.0.scale_shift_table_a2v_ca_video",
    "model.diffusion_model.av_ca_a2v_gate_adaln_single.linear.weight",
    "model.diffusion_model.video_embeddings_connector.learnable_registers",
    "model.diffusion_model.audio_embeddings_connector.learnable_registers",
    "text_embedding_projection.video_aggregate_embed.weight",
    "vae.decoder.conv_in.conv.weight",
    "vae.per_channel_statistics.mean-of-means",
    "audio_vae.decoder.conv_in.conv.weight",
    "vocoder.vocoder.conv_pre.weight",
]


class TensorRef(NamedTuple):
    """One output tensor, pointing at raw bytes in a source file."""

    out_key: str
    src_path: Path
    dtype: str
    shape: List[int]
    src_start: int  # absolute byte offset in the source file
    nbytes: int


def read_header(path: Path) -> Tuple[dict, int]:
    """Return (header_dict, data_section_start) for a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def remap_key(key: str, rules: List[Tuple[str, str]]) -> Optional[str]:
    """Apply the first matching prefix rule (longest first). None = drop."""
    for src_prefix, dst_prefix in sorted(rules, key=lambda r: -len(r[0])):
        if src_prefix == "":
            return dst_prefix + key
        if key.startswith(src_prefix):
            return dst_prefix + key[len(src_prefix) :]
    return None


def collect(src: Path, components: Iterable[Tuple[str, List[Tuple[str, str]]]]) -> List[TensorRef]:
    """Build the output tensor list from the diffusers subdirectories."""
    refs: List[TensorRef] = []
    seen: Dict[str, Path] = {}
    for subdir, rules in components:
        path = src / subdir / "model.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"missing component: {path}")
        header, data_start = read_header(path)
        header.pop("__metadata__", None)
        n_kept = 0
        for key, info in header.items():
            out_key = remap_key(key, rules)
            if out_key is None:
                print(f"    ! dropping unmapped key {subdir}/{key}")
                continue
            if out_key in seen:
                raise ValueError(f"key collision on {out_key!r}: {seen[out_key]} and {path}")
            seen[out_key] = path
            start, end = info["data_offsets"]
            refs.append(
                TensorRef(
                    out_key=out_key,
                    src_path=path,
                    dtype=info["dtype"],
                    shape=info["shape"],
                    src_start=data_start + start,
                    nbytes=end - start,
                )
            )
            n_kept += 1
        size_gb = sum(r.nbytes for r in refs if r.src_path == path) / 1e9
        print(f"  {subdir:28} {n_kept:5d} tensors  {size_gb:7.2f} GB")
    return refs


def write_safetensors(out_path: Path, refs: List[TensorRef], metadata: Dict[str, str]) -> None:
    """Stream ``refs`` into one safetensors file without materializing tensors.

    Tensor payloads are copied as raw bytes, so dtypes round-trip exactly and
    peak memory is one CHUNK regardless of model size.
    """
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for r in refs:
        header[r.out_key] = {"dtype": r.dtype, "shape": r.shape, "data_offsets": [offset, offset + r.nbytes]}
        offset += r.nbytes

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)  # 8-byte align the data section

    total = offset
    show_progress = sys.stdout.isatty()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    written = 0
    last_pct = -5
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        # Group by source file so each is opened once and read mostly forward.
        # Order within a group follows `refs`, which is also the write order,
        # so the output stays consistent with the header offsets above.
        by_src: Dict[Path, List[TensorRef]] = {}
        for r in refs:
            by_src.setdefault(r.src_path, []).append(r)
        for src_path, group in by_src.items():
            with open(src_path, "rb") as fin:
                for r in group:
                    fin.seek(r.src_start)
                    remaining = r.nbytes
                    while remaining:
                        buf = fin.read(min(CHUNK, remaining))
                        if not buf:
                            raise IOError(f"unexpected EOF reading {r.out_key} from {src_path}")
                        out.write(buf)
                        remaining -= len(buf)
                        written += len(buf)
                    if show_progress:
                        pct = int(100 * written / total) if total else 100
                        if pct >= last_pct + 5:
                            last_pct = pct - pct % 5
                            print(f"\r    writing {out_path.name}: {pct:3d}%  ({written / 1e9:.1f}/{total / 1e9:.1f} GB)", end="", flush=True)
    prefix = "\r" if show_progress else "    "
    print(f"{prefix}    writing {out_path.name}: done ({total / 1e9:.2f} GB)            ")
    tmp.replace(out_path)


def derive_vae_scale_factors(vae_cfg: dict) -> List[int]:
    """(time, height, width) downscale of the LTX video VAE, from its own config.

    Per ``video_vae.py``: ``patch_size`` sets the initial spatial patchify, then
    each ``compress_*`` block halves the axes it names -- ``compress_time*``
    temporal, ``compress_space*`` spatial, ``compress_all*`` both. ``multiplier``
    in those blocks is the CHANNEL multiplier and does not affect geometry.
    A stock LTX-2 VAE works out to [8, 32, 32] (H/32, W/32, 1 + (F-1)/8).

    This is derived rather than hardcoded because it is the one value that,
    if wrong, produces silently mis-shaped latents.
    """
    spatial = int(vae_cfg.get("patch_size", 4))
    temporal = 1
    for entry in vae_cfg.get("encoder_blocks", []):
        name = entry[0] if isinstance(entry, (list, tuple)) else entry
        if name.startswith("compress_time"):
            temporal *= 2
        elif name.startswith("compress_space"):
            spatial *= 2
        elif name.startswith("compress_all"):
            spatial *= 2
            temporal *= 2
    return [temporal, spatial, spatial]


def derive_audio_geometry(audio_cfg: dict) -> Dict[str, object]:
    """Audio latent geometry the LTX-2 runner needs, from the audio VAE's config.

    The runner sizes the audio latent as
    ``sampling_rate / hop_length / scale_factor`` frames by ``mel_bins`` bins,
    and reads each of those from the top-level config -- which a diffusers
    export keeps nested inside audio_vae/config.json instead.

    ``audio_scale_factor`` is the audio VAE's latent downsample, which LightX2V
    fixes at ``audio_vae.LATENT_DOWNSAMPLE_FACTOR = 4``; it also equals
    ``2 ** (len(ch_mult) - 1)`` for a stock config, so a disagreement means the
    VAE is not stock and is worth surfacing rather than silently trusting.

    ``audio_mel_bins`` is the LATENT bin count -- the mel channels after that
    downsample (64 / 4 = 16), not the 64 the VAE config lists.
    """
    pre = audio_cfg.get("preprocessing", {})
    params = audio_cfg.get("model", {}).get("params", {})
    ddconfig = params.get("ddconfig", {})

    sampling_rate = pre.get("audio", {}).get("sampling_rate") or params.get("sampling_rate") or 16000
    hop_length = pre.get("stft", {}).get("hop_length") or 160
    mel_channels = pre.get("mel", {}).get("n_mel_channels") or ddconfig.get("mel_bins") or 64

    scale_factor = 4  # LightX2V's audio_vae.LATENT_DOWNSAMPLE_FACTOR
    ch_mult = ddconfig.get("ch_mult")
    if ch_mult:
        implied = 2 ** (len(ch_mult) - 1)
        if implied != scale_factor:
            print(
                f"  note: audio VAE ch_mult={ch_mult} implies a {implied}x latent downsample, but LightX2V's audio decoder is fixed at {scale_factor}x. Emitting {scale_factor}; check the audio latent shape if output length looks wrong."
            )

    mel_bins, rem = divmod(int(mel_channels), scale_factor)
    if rem:
        print(f"  note: {mel_channels} mel channels is not divisible by the {scale_factor}x downsample; audio_mel_bins rounded down to {mel_bins}")

    return {
        "audio_sampling_rate": int(sampling_rate),
        "audio_hop_length": int(hop_length),
        "audio_scale_factor": int(scale_factor),
        "audio_mel_bins": int(mel_bins),
    }


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_main_metadata(src: Path) -> Dict[str, str]:
    """Assemble the ``config`` metadata blob the component loaders require."""
    merged: Dict[str, object] = {}
    for name in CONFIG_COMPONENTS:
        cfg = load_json(src / name / "config.json")
        # Upstream nests each payload under its own key and adds a sibling
        # diffusers ``_class_name``. Take only the payload, otherwise the
        # last component's _class_name would leak to the merged top level.
        merged[name] = cfg[name] if name in cfg else cfg
    # Not read by the LTX-2/2.3 path (which takes the DiT arch from
    # config.json on disk), but LTX-2.5 reads config["transformer"] from
    # metadata, and keeping it makes the file self-describing.
    merged["transformer"] = load_json(src / "transformer" / "config.json")
    return {"config": json.dumps(merged)}


def warn_if_stale_gemma_layout(gemma_dir: Path) -> None:
    """Flag a Gemma checkpoint whose keys predate the transformers VLM rename.

    transformers moved Gemma3's keys under a ``model.`` root during its VLM
    standardisation. An older export (``language_model.model.*``,
    ``vision_tower.vision_model.*``) loaded into a transformers that has dropped
    the compatibility shim produces no error -- every tensor is UNEXPECTED,
    every parameter is MISSING, and the text encoder is silently randomly
    initialised, so generation ignores the prompt. Worth catching here rather
    than in a load report nobody reads.
    """
    index = gemma_dir / "model.safetensors.index.json"
    if not index.exists():
        return
    try:
        with open(index) as f:
            keys = list(json.load(f)["weight_map"])
    except Exception:  # noqa: BLE001 - advisory only
        return
    if not keys or any(k.startswith("model.") for k in keys):
        return
    stale = sorted({k.split(".")[0] for k in keys})
    print(
        f"\n  WARNING: {gemma_dir} uses the pre-rename Gemma key layout (top-level: {stale}).\n"
        f"           Recent transformers expects everything under a 'model.' root. Loading as-is\n"
        f"           randomly initialises the text encoder WITHOUT failing, which looks like a\n"
        f"           working run that ignores the prompt. Check and fix with:\n"
        f"             python tools/convert/ltx2_remap_gemma.py --src {gemma_dir} --dry-run\n"
    )


def link_or_copy(src_dir: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        print(f"  gemma: {dst} already exists, leaving as-is")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src_dir.resolve(), dst)
        print(f"  gemma: symlinked {dst} -> {src_dir}")
    else:
        print(f"  gemma: copying {src_dir} -> {dst} (this is ~25 GB)")
        shutil.copytree(src_dir, dst)


def verify(out: Path, name: str) -> int:
    """Re-read the emitted files and sanity-check them. Returns exit status."""
    problems = 0
    main = out / f"{name}.safetensors"
    print(f"\nVerifying {out}")

    if not main.exists():
        print(f"  FAIL missing {main}")
        return 1

    header, _ = read_header(main)
    meta = header.pop("__metadata__", None)
    keys = set(header)
    print(f"  {main.name}: {len(keys)} tensors")

    if not meta or "config" not in meta:
        print("  FAIL main file has no __metadata__['config'] -- VAE/vocoder loaders will fail")
        problems += 1
    else:
        cfg = json.loads(meta["config"])
        missing_cfg = [k for k in ("vae", "audio_vae", "vocoder") if k not in cfg]
        if missing_cfg:
            print(f"  FAIL config metadata missing top-level keys: {missing_cfg}")
            problems += 1
        else:
            print(f"  config metadata: {sorted(cfg)}")

    for sentinel in DIT_SENTINEL_KEYS:
        if sentinel not in keys:
            print(f"  FAIL missing expected key: {sentinel}")
            problems += 1

    groups = {
        "model.diffusion_model.transformer_blocks.": 0,
        "model.diffusion_model.video_embeddings_connector.": 0,
        "model.diffusion_model.audio_embeddings_connector.": 0,
        "text_embedding_projection.": 0,
        "vae.": 0,
        "audio_vae.": 0,
        "vocoder.": 0,
    }
    for k in keys:
        for g in groups:
            if k.startswith(g):
                groups[g] += 1
                break
    print("  key groups:")
    for g, n in groups.items():
        flag = "  <-- EMPTY" if n == 0 else ""
        print(f"    {n:6d}  {g}{flag}")
        if n == 0:
            problems += 1

    for _, tag in UPSCALERS:
        up = out / f"{name}-{tag}.safetensors"
        if not up.exists():
            print(f"  note: {up.name} not present (fine if that ratio was not exported)")
            continue
        uh, _ = read_header(up)
        umeta = uh.pop("__metadata__", None)
        if not umeta or "config" not in umeta:
            print(f"  FAIL {up.name} has no __metadata__['config']")
            problems += 1
            continue
        ucfg = json.loads(umeta["config"])
        scale = ucfg.get("spatial_scale", 2.0) if ucfg.get("rational_resampler") else 2.0
        print(f"  {up.name}: {len(uh)} tensors, effective scale {scale}x")

    cfg_json = out / "config.json"
    if not cfg_json.exists():
        print(f"  FAIL missing {cfg_json} -- set_config needs it for the DiT architecture")
        problems += 1
    else:
        arch = load_json(cfg_json)
        print(f"  config.json: num_layers={arch.get('num_layers')} in_channels={arch.get('in_channels')} vae_scale_factors={arch.get('vae_scale_factors')}")
        if not arch.get("vae_scale_factors"):
            print("  FAIL config.json has no vae_scale_factors -- the runner sizes every video latent from it")
            problems += 1
        audio_keys = ["audio_sampling_rate", "audio_hop_length", "audio_scale_factor", "audio_mel_bins"]
        missing_audio = [k for k in audio_keys if arch.get(k) is None]
        if missing_audio:
            print(f"  FAIL config.json missing audio geometry: {missing_audio} -- the runner sizes the audio latent from these")
            problems += 1
        else:
            print(f"  audio geometry: {{{', '.join(f'{k}={arch[k]}' for k in audio_keys)}}}")

    gemma = out / "gemma"
    for probe in ("tokenizer.model", "preprocessor_config.json"):
        if not list(gemma.rglob(probe)):
            print(f"  FAIL gemma/ has no {probe}")
            problems += 1
    if not list(gemma.rglob("model*.safetensors")):
        print("  FAIL gemma/ has no model*.safetensors")
        problems += 1
    warn_if_stale_gemma_layout(gemma)

    print(f"\n{'PASS - conversion looks complete' if problems == 0 else f'{problems} PROBLEM(S) FOUND'}")
    return 0 if problems == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, help="diffusers-layout LTX-2.x repo root")
    p.add_argument("--out", type=Path, required=True, help="output directory (use as --model_path)")
    p.add_argument("--name", default="ltx-2.3", help="basename for the emitted files (default: ltx-2.3)")
    p.add_argument("--gemma", choices=["symlink", "copy", "skip"], default="symlink", help="how to expose text_encoder/gemma in the output dir (default: symlink)")
    p.add_argument("--skip-upscalers", action="store_true", help="do not emit the spatial upscaler files")
    p.add_argument("--verify-only", action="store_true", help="only re-check an existing output directory")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    args = p.parse_args()

    if args.verify_only:
        return verify(args.out, args.name)
    if args.src is None:
        p.error("--src is required unless --verify-only is given")
    if not args.src.is_dir():
        p.error(f"--src is not a directory: {args.src}")

    args.out.mkdir(parents=True, exist_ok=True)
    main_path = args.out / f"{args.name}.safetensors"
    if main_path.exists() and not args.overwrite:
        p.error(f"{main_path} exists; pass --overwrite to replace it")

    print(f"Reading diffusers components from {args.src}")
    refs = collect(args.src, MERGED_COMPONENTS)
    total_gb = sum(r.nbytes for r in refs) / 1e9
    print(f"  {'TOTAL':28} {len(refs):5d} tensors  {total_gb:7.2f} GB")

    print("\nAssembling config metadata")
    metadata = build_main_metadata(args.src)
    print(f"  config keys: {sorted(json.loads(metadata['config']))}")

    print(f"\nWriting merged checkpoint -> {main_path}")
    write_safetensors(main_path, refs, metadata)

    if not args.skip_upscalers:
        for subdir, tag in UPSCALERS:
            sub = args.src / subdir
            if not (sub / "model.safetensors").exists():
                print(f"\nSkipping {subdir}: not present in source")
                continue
            up_path = args.out / f"{args.name}-{tag}.safetensors"
            if up_path.exists() and not args.overwrite:
                print(f"\nSkipping {up_path.name}: exists (use --overwrite)")
                continue
            print(f"\nWriting upscaler -> {up_path}")
            up_refs = collect(args.src, [(subdir, [("", "")])])
            up_meta = {"config": json.dumps(load_json(sub / "config.json"))}
            write_safetensors(up_path, up_refs, up_meta)

    # set_config reads <model_path>/config.json and merges it AFTER the user's
    # --config_json, so this file is authoritative. The diffusers export splits
    # its configs per component, which means the transformer config alone is
    # missing the pipeline-level geometry the runner indexes directly --
    # notably vae_scale_factors, whose absence is a hard KeyError in
    # get_latent_shape_with_target_hw. Fold it in here.
    arch = load_json(args.src / "transformer" / "config.json")
    vae_cfg = load_json(args.src / "vae" / "config.json")
    vae_cfg = vae_cfg.get("vae", vae_cfg)
    scale_factors = derive_vae_scale_factors(vae_cfg)
    arch["vae_scale_factors"] = scale_factors
    arch.setdefault("vae_stride", scale_factors)

    audio_cfg = load_json(args.src / "audio_vae" / "config.json")
    audio_geom = derive_audio_geometry(audio_cfg.get("audio_vae", audio_cfg))
    arch.update(audio_geom)
    with open(args.out / "config.json", "w") as f:
        json.dump(arch, f, indent=2)
    print(f"\nWrote {args.out / 'config.json'} (DiT architecture, num_layers={arch.get('num_layers')}, vae_scale_factors={scale_factors})")
    print(f"  audio geometry: {audio_geom}")
    if scale_factors != [8, 32, 32]:
        print(f"  note: derived {scale_factors} rather than the stock LTX-2 [8, 32, 32] -- double-check the VAE's encoder_blocks if that is unexpected")

    if args.gemma != "skip":
        gemma_src = args.src / "text_encoder" / "gemma"
        link_or_copy(gemma_src, args.out / "gemma", args.gemma)
        warn_if_stale_gemma_layout(gemma_src)

    # A ready-to-run LightX2V config. gemma_original_ckpt is set explicitly
    # because the default resolution rglobs model_path for model*.safetensors,
    # which is ambiguous once other checkpoints live alongside it.
    run_cfg = {
        "_comment": f"Generated by ltx2_diffusers_to_lightx2v.py from {args.src}. Use with --model_path {args.out}.",
        "infer_steps": 8,
        "target_video_length": 241,
        "target_height": 1024,
        "target_width": 1536,
        # flash_attn4 spans every GPU this branch targets (Hopper + Blackwell),
        # whereas sage_attn2 has no sm100/sm103 kernels and would not load on
        # B200/B300. Batch size is 1 and sequences are packed, so FA4's
        # bs==1 / no-varlen restriction does not bite here.
        "attn_type": "flash_attn4",
        "sample_guide_scale": 1,
        "sample_shift": [2.05, 0.95],
        "enable_cfg": False,
        "cpu_offload": True,
        "offload_granularity": "block",
        "num_channels_latents": 128,
        "fps": 24,
        "audio_fps": 24000,
        "audio_mel_bins": 16,
        "double_precision_rope": True,
        "caption_proj_before_connector": True,
        "cross_attention_adaln": True,
        "apply_gated_attention": True,
        "dit_original_ckpt": str(main_path.resolve()),
        "gemma_original_ckpt": str((args.out / "gemma").resolve()),
        "distilled_sigma_values": [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
        "use_upsampler": True,
        "upsampler_original_ckpt": str((args.out / f"{args.name}-spatial-upscaler-x2.safetensors").resolve()),
        "distilled_sigma_values_upsample": [0.909375, 0.725, 0.421875, 0.0],
    }
    run_cfg_path = args.out / "lightx2v_ltx2_3.json"
    with open(run_cfg_path, "w") as f:
        json.dump(run_cfg, f, indent=4)
    print(f"Wrote {run_cfg_path} (ready-to-run LightX2V config)")

    status = verify(args.out, args.name)
    if status == 0:
        print(
            "\nRun with:\n"
            f"  python -m lightx2v.infer --model_cls ltx2 --task t2av \\\n"
            f"      --model_path {args.out} \\\n"
            f"      --config_json {run_cfg_path} \\\n"
            f'      --prompt "..." --save_result_path out.mp4'
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
