#!/usr/bin/env python3
"""Remap an LTX-2 Gemma text-encoder checkpoint onto the installed transformers' key layout.

Why this exists
---------------
transformers renamed Gemma3's state-dict keys during its VLM standardisation,
and kept moving them afterwards. Three layouts seen in the wild:

    checkpoint (older export)   language_model.model.*      vision_tower.vision_model.*
    transformers 5.5            model.language_model.*      model.vision_tower.vision_model.*
    transformers 5.15           model.language_model.*      model.vision_tower.*

Load an old-layout checkpoint into a new transformers and every tensor lands in
the UNEXPECTED column while every parameter the model wants lands in MISSING --
so transformers randomly initialises the whole text encoder and inference
silently produces prompt-independent garbage. The give-away in the log is:

    This checkpoint seem corrupted. The tied weights mapping ... specifies to tie
    model.language_model.embed_tokens.weight to lm_head.weight, but both are
    absent from the checkpoint

Downgrading transformers is not a way out: LightX2V's own Gemma code imports
``transformers.masking_utils``, which arrived in the same refactor that renamed
the keys. So the checkpoint is what has to move.

How the mapping is chosen
-------------------------
It is DERIVED, never hardcoded, because the target layout depends on the
installed transformers version (see the table above). The script instantiates
``Gemma3ForConditionalGeneration`` on the meta device from the checkpoint's own
config, reads the key set the model actually wants, and matches each checkpoint
key to it by longest component-boundary suffix, requiring a unique hit. It then
refuses to write anything unless every model parameter is accounted for.

That means running this on the machine that will run inference: it adapts to
whatever transformers is installed there.

Memory: tensor payloads are copied as raw bytes, so peak RSS is one chunk
regardless of model size.

Usage:
    # inspect the derived mapping without writing (do this first)
    python tools/convert/ltx2_remap_gemma.py --src .../text_encoder/gemma --dry-run

    # write the remapped checkpoint
    python tools/convert/ltx2_remap_gemma.py \\
        --src .../text_encoder/gemma --out .../gemma_remapped

    # then point the LightX2V config at it
    "gemma_original_ckpt": ".../gemma_remapped"
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

CHUNK = 32 * 1024 * 1024

# Files that must travel with the weights for from_pretrained / the tokenizer
# and processor lookups LightX2V does (_find_matching_dir searches for
# tokenizer.model and preprocessor_config.json).
SIDECAR_GLOBS = ["*.json", "*.model", "*.txt"]


def read_header(path: Path) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def model_key_set(src: Path) -> Tuple[Set[str], str]:
    """Key set that the locally installed transformers' Gemma3 model expects."""
    try:
        import torch
        import transformers
        from transformers import Gemma3ForConditionalGeneration
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"need torch + transformers installed to derive the mapping: {e}")

    cfg_path = src / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json in {src}")
    with open(cfg_path) as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if k not in ("architectures", "_name_or_path")}
    cfg = transformers.Gemma3Config(**raw)
    with torch.device("meta"):
        model = Gemma3ForConditionalGeneration(cfg)
    return set(model.state_dict().keys()), transformers.__version__


def checkpoint_keys(src: Path) -> Dict[str, str]:
    """{tensor_key: shard_filename} for a sharded or single-file checkpoint."""
    index = src / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            return dict(json.load(f)["weight_map"])
    single = src / "model.safetensors"
    if single.exists():
        header, _ = read_header(single)
        header.pop("__metadata__", None)
        return {k: single.name for k in header}
    raise SystemExit(f"no model.safetensors(.index.json) in {src}")


def derive_mapping(have: List[str], want: Set[str]) -> Tuple[Dict[str, str], List[str]]:
    """Map each checkpoint key onto a model key, via derived prefix rules.

    Two passes, because suffix matching alone is not enough: the language model
    and the vision tower share tails like ``layers.0.self_attn.q_proj.weight``,
    and the component that tells them apart (``language_model`` vs
    ``vision_tower``) sits at a different depth in each name.

    Pass 1 matches only the keys whose longest component-boundary suffix is
    UNIQUE in the model's key set -- e.g. ``layers.0.input_layernorm.weight``,
    which only the language model has. Each such match implies one
    (source prefix -> target prefix) rule.

    Pass 2 applies those rules, longest source prefix first, to every key. The
    caller then checks the result covers the model exactly, so a wrong rule
    cannot slip through silently.
    """
    by_suffix: Dict[str, List[str]] = {}
    for w in want:
        parts = w.split(".")
        for i in range(len(parts)):
            by_suffix.setdefault(".".join(parts[i:]), []).append(w)

    # ---- pass 1: learn prefix rules from unambiguous keys ----
    rules: Dict[str, str] = {}
    for k in have:
        if k in want:
            continue
        parts = k.split(".")
        for i in range(len(parts)):
            cands = by_suffix.get(".".join(parts[i:]))
            if not cands or len(cands) != 1:
                continue
            hit = cands[0]
            tail_len = len(parts) - i
            src_prefix = ".".join(parts[: len(parts) - tail_len])
            hit_parts = hit.split(".")
            dst_prefix = ".".join(hit_parts[: len(hit_parts) - tail_len])
            if src_prefix != dst_prefix:
                rules.setdefault(src_prefix, dst_prefix)
            break

    # ---- pass 2: apply them ----
    ordered = sorted(rules.items(), key=lambda kv: -len(kv[0]))
    mapping: Dict[str, str] = {}
    problems: List[str] = []
    for k in have:
        if k in want:
            mapping[k] = k
            continue
        target = None
        for src_prefix, dst_prefix in ordered:
            # An empty source prefix is a real rule, not a bug: the vision tower
            # and multi_modal_projector are only re-parented ("" -> "model"),
            # their own names unchanged. It has to be matched explicitly, since
            # startswith("." ) never fires.
            if src_prefix == "":
                target = f"{dst_prefix}.{k}" if dst_prefix else k
                break
            if k == src_prefix or k.startswith(src_prefix + "."):
                rest = k[len(src_prefix) :].lstrip(".")
                target = f"{dst_prefix}.{rest}" if dst_prefix else rest
                break
        if target is None:
            problems.append(f"{k}: no derived prefix rule applies")
        elif target not in want:
            problems.append(f"{k}: mapped to {target!r}, which the model does not have")
        else:
            mapping[k] = target
    return mapping, problems


def write_shard(out_path: Path, src_path: Path, keys: List[Tuple[str, str]], metadata: Optional[dict]) -> int:
    """Copy one shard, renaming keys. `keys` is [(src_key, dst_key), ...]."""
    header, data_start = read_header(src_path)
    header.pop("__metadata__", None)

    out_header: Dict[str, object] = {}
    if metadata:
        out_header["__metadata__"] = metadata
    plan = []
    offset = 0
    for src_key, dst_key in keys:
        info = header[src_key]
        start, end = info["data_offsets"]
        nbytes = end - start
        out_header[dst_key] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + nbytes]}
        plan.append((data_start + start, nbytes))
        offset += nbytes

    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with open(tmp, "wb") as out, open(src_path, "rb") as fin:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for src_off, nbytes in plan:
            fin.seek(src_off)
            remaining = nbytes
            while remaining:
                buf = fin.read(min(CHUNK, remaining))
                if not buf:
                    raise IOError(f"unexpected EOF in {src_path}")
                out.write(buf)
                remaining -= len(buf)
    tmp.replace(out_path)
    return offset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Gemma directory to read (…/text_encoder/gemma)")
    p.add_argument("--out", type=Path, help="directory to write; required unless --dry-run")
    p.add_argument("--dry-run", action="store_true", help="derive and print the mapping, write nothing")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.dry_run and args.out is None:
        p.error("--out is required unless --dry-run is given")
    if not args.src.is_dir():
        p.error(f"--src is not a directory: {args.src}")

    want, tf_ver = model_key_set(args.src)
    weight_map = checkpoint_keys(args.src)
    have = sorted(weight_map)
    print(f"transformers {tf_ver} expects {len(want)} tensors; checkpoint has {len(have)}")

    mapping, problems = derive_mapping(have, want)

    # Group the derived renames into readable prefix rules.
    rules: Dict[Tuple[str, str], int] = {}
    for src_key, dst_key in mapping.items():
        sp, dp = src_key.split("."), dst_key.split(".")
        i = 0
        while i < min(len(sp), len(dp)) and sp[len(sp) - 1 - i] == dp[len(dp) - 1 - i]:
            i += 1
        rules[(".".join(sp[: len(sp) - i]), ".".join(dp[: len(dp) - i]))] = rules.get((".".join(sp[: len(sp) - i]), ".".join(dp[: len(dp) - i])), 0) + 1
    print("\nderived prefix rules:")
    for (src_pre, dst_pre), n in sorted(rules.items(), key=lambda kv: -kv[1]):
        arrow = f"{src_pre or '(root)'} -> {dst_pre or '(root)'}"
        print(f"  {n:5d}  {arrow}{'   [unchanged]' if src_pre == dst_pre else ''}")

    unchanged = sum(n for (a, b), n in rules.items() if a == b)
    print(f"\n  renamed {len(mapping) - unchanged}, already correct {unchanged}")

    if problems:
        print(f"\n{len(problems)} key(s) could not be mapped:")
        for pr in problems[:10]:
            print("   ", pr)
        if len(problems) > 10:
            print(f"    ... (+{len(problems) - 10} more)")
        print("\nRefusing to write a partially-mapped checkpoint.")
        return 1

    # Every parameter the model wants must be produced, except weights it ties
    # (lm_head is tied to embed_tokens in Gemma3, so it is legitimately absent).
    produced = set(mapping.values())
    unfilled = sorted(want - produced)
    tied_ok = {k for k in unfilled if k.endswith("lm_head.weight")}
    real_gaps = [k for k in unfilled if k not in tied_ok]
    if tied_ok:
        print(f"  not written, expected to be tied: {sorted(tied_ok)}")
    if real_gaps:
        print(f"\n{len(real_gaps)} model parameter(s) would still be missing, e.g.:")
        for k in real_gaps[:10]:
            print("   ", k)
        print("\nRefusing to write -- this would leave randomly-initialised weights.")
        return 1
    if len(produced) != len(mapping):
        print(f"\nCollision: {len(mapping)} source keys collapsed onto {len(produced)} targets. Refusing to write.")
        return 1
    print("  every model parameter accounted for")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if args.out.exists() and not args.overwrite:
        p.error(f"{args.out} exists; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    # Rewrite each shard, preserving the original sharding.
    by_shard: Dict[str, List[Tuple[str, str]]] = {}
    for src_key in have:
        by_shard.setdefault(weight_map[src_key], []).append((src_key, mapping[src_key]))

    print(f"\nwriting {len(by_shard)} shard(s) -> {args.out}")
    new_weight_map: Dict[str, str] = {}
    total = 0
    for shard, keys in sorted(by_shard.items()):
        src_shard = args.src / shard
        hdr, _ = read_header(src_shard)
        meta = hdr.get("__metadata__")
        n = write_shard(args.out / shard, src_shard, keys, meta)
        total += n
        for _, dst_key in keys:
            new_weight_map[dst_key] = shard
        print(f"  {shard}: {len(keys):5d} tensors, {n / 1e9:.2f} GB")

    if (args.src / "model.safetensors.index.json").exists():
        with open(args.src / "model.safetensors.index.json") as f:
            index = json.load(f)
        index["weight_map"] = new_weight_map
        with open(args.out / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        print("  rewrote model.safetensors.index.json")

    copied = 0
    for pattern in SIDECAR_GLOBS:
        for f in args.src.glob(pattern):
            if f.name == "model.safetensors.index.json":
                continue
            shutil.copy2(f, args.out / f.name)
            copied += 1
    print(f"  copied {copied} config/tokenizer file(s)")
    print(f"\ndone: {total / 1e9:.2f} GB written to {args.out}")
    print(f'\nPoint your LightX2V config at it:\n  "gemma_original_ckpt": "{args.out.resolve()}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
