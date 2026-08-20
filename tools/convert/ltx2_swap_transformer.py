#!/usr/bin/env python3
"""Swap the transformer weights of an LTX-2.x all-in-one checkpoint.

The debugging move this enables: keep everything from an OFFICIAL release --
video VAE, audio VAE, vocoder, text-embedding projection, and the metadata,
all known-good -- and replace only the ``model.diffusion_model.*`` tensors
with a LoRA-merged transformer exported from ComfyUI. If artifacts persist
with official audio/VAE weights, the transformer (or the runtime) is the
culprit; if they vanish, the exported repo's other components were damaged.

Only the transformer namespace is ever replaced. Export keys outside it
(a ComfyUI checkpoint-save may bundle vae./text encoder tensors) are ignored
and reported, never merged. Base tensors the export does not cover (e.g. the
embeddings connectors, which some export nodes omit) are kept from the base --
correct here, since the LoRAs only touch transformer blocks.

Key prefixes are auto-detected: exports have been seen as
``model.diffusion_model.*`` (checkpoint-style), ``diffusion_model.*``
(model-save style), and bare diffusers names (``transformer_blocks.*``). The
prefix is chosen by which one actually lands on the base's keys, and the
result is refused unless it covers the export.

Guards, all hard errors unless stated:
  * shape mismatch between a replaced tensor and the base's -- always fatal;
  * export containing fp8/int tensors or a ``scaled_fp8`` marker: a scaled-fp8
    ComfyUI export cannot be merged raw (its scales would be dropped and the
    weights are not plain bf16) -- re-export in bf16;
  * dtype changes among float types are allowed but reported.

Tensor payloads stream as raw bytes (fixed-size chunks), so a 44 GB merge
runs in constant memory and replaced bytes are bit-exact copies of the export.

Usage:
    # inspect what would happen
    python tools/convert/ltx2_swap_transformer.py \\
        --base ltx-2.3-22b-dev.safetensors \\
        --transformer comfy_merged_transformer.safetensors --dry-run

    # write the merged checkpoint
    python tools/convert/ltx2_swap_transformer.py \\
        --base ltx-2.3-22b-dev.safetensors \\
        --transformer comfy_merged_transformer.safetensors \\
        --out 10eros-v1-official-base.safetensors

    # then point the config's dit_original_ckpt at the output
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

CHUNK = 32 * 1024 * 1024
DIT_NS = "model.diffusion_model."
PREFIX_CANDIDATES = ["", "model.", DIT_NS]
FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}


def read_header(path: Path) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def detect_prefix(export_keys: List[str], base_keys: set) -> Tuple[str, int]:
    """Prefix that, prepended to export keys, lands them on the base's DiT keys."""
    best, best_hits = None, -1
    for pre in PREFIX_CANDIDATES:
        hits = sum(1 for k in export_keys if (pre + k).startswith(DIT_NS) and (pre + k) in base_keys)
        if hits > best_hits:
            best, best_hits = pre, hits
    return best, best_hits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="official all-in-one checkpoint (metadata and non-transformer weights come from here)")
    p.add_argument("--transformer", type=Path, required=True, help="transformer-only export (e.g. ComfyUI LoRA-merged save)")
    p.add_argument("--out", type=Path, help="output file; required unless --dry-run")
    p.add_argument("--dry-run", action="store_true", help="report the merge plan without writing")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.dry_run and args.out is None:
        p.error("--out is required unless --dry-run is given")
    for f in (args.base, args.transformer):
        if not f.is_file():
            p.error(f"not a file: {f}")

    base_header, _ = read_header(args.base)
    base_meta = base_header.pop("__metadata__", None)
    export_header, export_data_start = read_header(args.transformer)
    export_header.pop("__metadata__", None)

    # ---- refuse quantized exports outright ----
    if "scaled_fp8" in export_header:
        print("ERROR: the export carries a 'scaled_fp8' marker -- it is a ComfyUI scaled-fp8 save. Merging it raw would drop the scales. Re-export the merged model in bf16.")
        return 1
    non_float = {k: v["dtype"] for k, v in export_header.items() if v["dtype"] not in FLOAT_DTYPES}
    if non_float:
        sample = list(non_float.items())[:5]
        print(f"ERROR: {len(non_float)} export tensor(s) are not plain float (e.g. {sample}). A quantized export cannot be merged raw; re-export in bf16.")
        return 1

    base_keys = set(base_header)
    base_dit = {k for k in base_keys if k.startswith(DIT_NS)}
    export_keys = list(export_header)

    prefix, hits = detect_prefix(export_keys, base_keys)
    if hits == 0:
        print("ERROR: no key prefix maps the export onto the base's transformer namespace.")
        print(f"  export samples: {sorted(export_keys)[:4]}")
        print(f"  base DiT samples: {sorted(base_dit)[:4]}")
        return 1
    print(f"prefix: {prefix!r} maps {hits}/{len(export_keys)} export keys onto the base's transformer")

    # ---- build the replacement plan ----
    replaced: Dict[str, str] = {}  # base key -> export key
    ignored: List[str] = []
    for k in export_keys:
        bk = prefix + k
        if bk.startswith(DIT_NS) and bk in base_keys:
            replaced[bk] = k
        else:
            ignored.append(k)
    kept_dit = sorted(base_dit - set(replaced))

    print(f"\nplan against {args.base.name} ({len(base_keys)} tensors):")
    print(f"  transformer tensors replaced : {len(replaced)}/{len(base_dit)}")
    print(f"  transformer tensors kept     : {len(kept_dit)}  (from base)")
    if kept_dit:
        from collections import Counter

        groups = Counter(".".join(k.split(".")[2:4]) for k in kept_dit)
        for g, n in sorted(groups.items(), key=lambda kv: -kv[1])[:6]:
            print(f"      {n:5d}  model.diffusion_model.{g}*")
    if ignored:
        from collections import Counter

        groups = Counter(k.split(".")[0] for k in ignored)
        print(f"  export keys ignored          : {len(ignored)}  (outside the transformer namespace, or unknown to the base)")
        for g, n in sorted(groups.items(), key=lambda kv: -kv[1])[:6]:
            print(f"      {n:5d}  {g}.*")
    other = len(base_keys) - len(base_dit)
    print(f"  non-transformer tensors      : {other}  (vae/audio_vae/vocoder/text_embedding_projection, all from base)")
    print(f"  metadata                     : copied verbatim from base ({sorted(base_meta) if base_meta else None})")

    # ---- shape / dtype checks ----
    mismatched = [(bk, base_header[bk]["shape"], export_header[ek]["shape"]) for bk, ek in replaced.items() if base_header[bk]["shape"] != export_header[ek]["shape"]]
    if mismatched:
        print(f"\nERROR: {len(mismatched)} replaced tensor(s) change shape -- the export does not match this base's architecture:")
        for bk, bs, es in mismatched[:8]:
            print(f"    {bk}: base {bs} vs export {es}")
        return 1
    dtype_changed = [(bk, base_header[bk]["dtype"], export_header[ek]["dtype"]) for bk, ek in replaced.items() if base_header[bk]["dtype"] != export_header[ek]["dtype"]]
    if dtype_changed:
        kinds = {(a, b) for _, a, b in dtype_changed}
        print(f"\n  note: {len(dtype_changed)} replaced tensor(s) change float dtype ({kinds}); the loader casts float weights to the inference dtype, so this is tolerated.")

    if len(replaced) < len(base_dit):
        print(
            f"\n  note: {len(base_dit) - len(replaced)} transformer tensor(s) stay at base values. Fine when the export omits parts the LoRAs never touched (connectors); NOT fine if the export was supposed to be complete."
        )

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if args.out.exists() and not args.overwrite:
        p.error(f"{args.out} exists; pass --overwrite")

    # ---- rebuild the header in the base's key order ----
    out_header: Dict[str, object] = {}
    if base_meta:
        out_header["__metadata__"] = base_meta
    plan: List[Tuple[Path, int, int]] = []  # (source file, absolute offset, nbytes)
    base_hdr_full, base_data_start = read_header(args.base)
    offset = 0
    for key, info in base_header.items():
        if key in replaced:
            src_info = export_header[replaced[key]]
            src_start, src_end = src_info["data_offsets"]
            plan.append((args.transformer, export_data_start + src_start, src_end - src_start))
            out_header[key] = {"dtype": src_info["dtype"], "shape": src_info["shape"], "data_offsets": [offset, offset + (src_end - src_start)]}
            offset += src_end - src_start
        else:
            src_start, src_end = info["data_offsets"]
            plan.append((args.base, base_data_start + src_start, src_end - src_start))
            out_header[key] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + (src_end - src_start)]}
            offset += src_end - src_start

    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)

    print(f"\nwriting {args.out} ({offset / 1e9:.2f} GB)")
    tmp = args.out.with_suffix(args.out.suffix + ".part")
    written, last_pct = 0, -5
    with open(tmp, "wb") as out, open(args.base, "rb") as fbase, open(args.transformer, "rb") as fexp:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        handles = {args.base: fbase, args.transformer: fexp}
        for src, start, nbytes in plan:
            f = handles[src]
            f.seek(start)
            remaining = nbytes
            while remaining:
                buf = f.read(min(CHUNK, remaining))
                if not buf:
                    raise IOError(f"unexpected EOF in {src}")
                out.write(buf)
                remaining -= len(buf)
                written += len(buf)
            pct = int(100 * written / offset) if offset else 100
            if pct >= last_pct + 5:
                last_pct = pct - pct % 5
                print(f"\r  {pct:3d}%  ({written / 1e9:.1f}/{offset / 1e9:.1f} GB)", end="", flush=True)
    print()
    tmp.replace(args.out)
    print(f"done. Point the config's dit_original_ckpt at {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
