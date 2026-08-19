#!/usr/bin/env bash
#
# One-shot LightX2V installer.
#
#   ./install.sh                      # core deps + the operators that fit this GPU
#   ./install.sh --minimal            # pure-Python only, no compilation (torch_sdpa attention)
#   ./install.sh --dry-run            # print every command without running it
#   ./install.sh --attn fa4,sage2     # pick attention backends explicitly
#   ./install.sh --quant sgl,vllm     # pick quantization backends explicitly
#   ./install.sh --arch 10.0          # override the detected compute capability
#   ./install.sh --verify             # only report what is already installed
#
# Scope: server GPUs only -- H100/H200 (sm90), B200 (sm100), B300 (sm103).
# Consumer Blackwell (5090, RTX Pro 6000 / sm120) is deliberately out of scope
# on this branch, which also means the nvfp4 / mxfp* path is out of scope:
# lightx2v_kernel compiles for sm120a only.
#
# Why this is not a single `pip install -e .`: the Python dependency set is only
# half the story. Attention and quantization operators are compiled from source
# with architecture-specific flags, and which ones are even buildable depends on
# the GPU. This script picks a working set for the detected hardware and tells
# you what it skipped and why.
#
# Everything the script installs beyond the core is optional at runtime, with
# one exception: RMSNorm falls back to pure torch when sgl-kernel is absent,
# but attention does NOT degrade -- the registry hands back a None kernel and
# the call fails. So whatever attn_type your config names has to be installed.
#
# Configs added on this branch use attn_type=flash_attn4, which works on every
# GPU here. Upstream's stock LTX-2 configs still say sage_attn2, which will not
# load on B200/B300; switch those to flash_attn4, or to torch_sdpa if you ran
# with --minimal and have no operators at all.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${LIGHTX2V_BUILD_DIR:-${REPO_ROOT}/.build-deps}"

MINIMAL=0
DRY_RUN=0
VERIFY_ONLY=0
ATTN_SPEC=""
QUANT_SPEC=""
ARCH_OVERRIDE=""
PIP="${PIP:-pip}"
JOBS="${MAX_JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) )}"

SKIPPED=()
INSTALLED=()

# ----------------------------------------------------------------------------
# plumbing
# ----------------------------------------------------------------------------

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '    \033[2m$ %s\033[0m\n' "$*"
    else
        printf '    \033[2m$ %s\033[0m\n' "$*"
        "$@"
    fi
}

# Run a command string through bash -c (for pipelines / env-prefixed builds).
run_sh() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '    \033[2m$ %s\033[0m\n' "$1"
    else
        printf '    \033[2m$ %s\033[0m\n' "$1"
        bash -c "$1"
    fi
}

skip() { printf '\033[1;33m[skip]\033[0m %s\n' "$1"; SKIPPED+=("$1"); }
ok()   { INSTALLED+=("$1"); }

# Clone (or reuse) a source dependency under BUILD_DIR.
fetch() {
    local url="$1" dir="$2" extra="${3:-}"
    if [[ -d "${BUILD_DIR}/${dir}/.git" ]]; then
        log "reusing existing clone ${BUILD_DIR}/${dir}"
        return 0
    fi
    run mkdir -p "${BUILD_DIR}"
    run_sh "git clone ${extra} '${url}' '${BUILD_DIR}/${dir}'"
}

has_python_module() {
    python3 -c "import $1" >/dev/null 2>&1
}

# ----------------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --minimal)  MINIMAL=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --verify)   VERIFY_ONLY=1; shift ;;
        --attn)     ATTN_SPEC="$2"; shift 2 ;;
        --quant)    QUANT_SPEC="$2"; shift 2 ;;
        --arch)     ARCH_OVERRIDE="$2"; shift 2 ;;
        -h|--help)  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          die "unknown option: $1 (try --help)" ;;
    esac
done

# ----------------------------------------------------------------------------
# environment detection
# ----------------------------------------------------------------------------

detect_env() {
    PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"

    TORCH_VER=""
    TORCH_CUDA=""
    if has_python_module torch; then
        TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
        TORCH_CUDA="$(python3 -c 'import torch; print(torch.version.cuda or "")')"
    fi

    NVCC_CUDA=""
    if command -v nvcc >/dev/null 2>&1; then
        # NB: [0-9][0-9]* rather than [0-9]\+ -- BSD sed rejects \+ in a BRE
        # and would silently yield an empty version, which would then skip the
        # cu13 wheel selection below.
        NVCC_CUDA="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
    fi

    ARCH=""
    GPU_NAME=""
    if [[ -n "${ARCH_OVERRIDE}" ]]; then
        ARCH="${ARCH_OVERRIDE}"
        GPU_NAME="(from --arch)"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    fi

    log "environment"
    echo "    python           : ${PY_VER}"
    echo "    torch            : ${TORCH_VER:-<not installed>}  (cuda ${TORCH_CUDA:-n/a})"
    echo "    nvcc             : ${NVCC_CUDA:-<not found>}"
    echo "    gpu              : ${GPU_NAME:-<none detected>}  (compute cap ${ARCH:-unknown})"
    echo "    build dir        : ${BUILD_DIR}"
    echo "    parallel jobs    : ${JOBS}"

    case "${PY_VER}" in
        3.1[0-9]) : ;;
        unknown)  die "python3 not found on PATH" ;;
        *)        die "LightX2V needs Python >= 3.10, found ${PY_VER}" ;;
    esac

    case "${ARCH}" in
        9.0)        GPU_CLASS="hopper" ;;
        10.0|10.3)  GPU_CLASS="blackwell-server" ;;
        "")         GPU_CLASS="unknown"
                    warn "no GPU detected; defaulting to the Hopper operator set. Pass --arch 9.0 / 10.0 / 10.3 to target explicitly." ;;
        12.0)       die "compute cap 12.0 (5090 / RTX Pro 6000) is out of scope on this branch; it targets H100/B200/B300. Use upstream's dockerfiles/Dockerfile_5090 for consumer Blackwell." ;;
        *)          die "compute cap ${ARCH} is not a supported server GPU on this branch (expected 9.0 for H100/H200, 10.0 for B200, 10.3 for B300)." ;;
    esac
    echo "    gpu class        : ${GPU_CLASS}"

    # nvcc version only matters for the operators built from source below
    # (SageAttention 3, SpargeAttn, MagiAttention). The torch wheel carries its
    # own CUDA runtime and is chosen separately.
    if [[ "${GPU_CLASS}" == "blackwell-server" ]]; then
        case "${NVCC_CUDA}" in
            13.*) : ;;
            "")   warn "no nvcc on PATH. Source-built operators (sage3) need a CUDA 13.0 toolkit on B200/B300; install one or use the prebuilt image: lightx2v/lightx2v:<date>-cu130" ;;
            *)    warn "nvcc reports CUDA ${NVCC_CUDA}, but B200/B300 want CUDA 13.0 -- that is what upstream builds and tests (dockerfiles/Dockerfile_cu130: torch 2.11 + cuda 13.0, and the docs recommend cuda130 for speed). Source-built operators will compile against ${NVCC_CUDA} and may not emit sm100/sm103 code. Prebuilt alternative: lightx2v/lightx2v:<date>-cu130" ;;
        esac
    fi
}

# Which attention backends make sense for this compute capability.
#
# The hard constraints, read off the upstream build recipes in dockerfiles/:
#   * SageAttention 2 is compiled for 8.0/8.6/8.9/9.0/12.0. sm90 is in that
#     list, sm100/sm103 are NOT -- so sage_attn2 works on H100 and is simply
#     unavailable on B200/B300.
#   * SageAttention 3 (sageattn3_blackwell) is the Blackwell FP4 attention:
#     B200/B300 only, not Hopper.
#   * FlashAttention 4 (CuTe DSL) covers Hopper AND Blackwell, ships as a
#     published wheel, and JITs its kernels -- so it is the default everywhere
#     here and FlashAttention 2/3 are not installed at all. FA2 in particular
#     cost a 20-60 min source build for a strictly older kernel.
#   * SpargeAttn tops out at 9.0, so it is Hopper-only here.
default_attn_for_arch() {
    case "${1:-}" in
        9.0)        echo "fa4,sage2" ;;                  # H100/H200
        10.0|10.3)  echo "fa4,sage3" ;;                  # B200/B300: no sage2
        *)          echo "fa4" ;;                        # unknown: the portable one
    esac
}

# sgl-kernel is the right default on every supported arch: it backs the
# fp8-sgl / int8-sgl GEMM schemes AND provides the fused RMSNorm that LTX-2
# selects by default (rms_norm_type "sgl-kernel"), which otherwise falls back
# to a slower pure-torch path. Everything else is opt-in via --quant.
default_quant_for_arch() {
    echo "sgl"
}

# ----------------------------------------------------------------------------
# core python dependencies
# ----------------------------------------------------------------------------

install_core() {
    log "core Python dependencies"

    if [[ -z "${TORCH_VER}" ]]; then
        local backend="${UV_TORCH_BACKEND:-}"
        if [[ -z "${backend}" ]]; then
            # Blackwell server parts are a cu13 target: upstream's reference
            # image is Dockerfile_cu130 (torch 2.11 + CUDA 13.0) and the docs
            # recommend cuda130 for speed. A pip torch wheel bundles its own
            # CUDA runtime, so this index does NOT have to match the local
            # nvcc -- only the driver has to be new enough. nvcc still matters
            # for the source-built operators below, which is a separate check.
            if [[ "${GPU_CLASS}" == "blackwell-server" ]]; then
                backend="cu130"
            else
                case "${NVCC_CUDA}" in
                    13.*)      backend="cu130" ;;
                    12.8|12.9) backend="cu128" ;;
                    12.*)      backend="cu126" ;;
                esac
            fi
        fi
        if [[ -n "${backend}" ]]; then
            log "installing torch for ${backend} (override with UV_TORCH_BACKEND)"
            run_sh "${PIP} install 'torch<=2.11.0' 'torchvision<=0.26.0' 'torchaudio<=2.11.0' --index-url https://download.pytorch.org/whl/${backend}"
        else
            warn "could not infer a CUDA wheel index; installing default torch build"
            run_sh "${PIP} install 'torch<=2.11.0' 'torchvision<=0.26.0' 'torchaudio<=2.11.0'"
        fi
    else
        log "torch ${TORCH_VER} already present, leaving it alone"
        # pyproject pins torch<=2.11.0; a newer one means the editable install
        # below will try to downgrade it out from under you.
        local major_minor="${TORCH_VER%%+*}"
        if [[ "$(printf '%s\n2.11.0\n' "${major_minor}" | sort -V | tail -1)" != "2.11.0" ]]; then
            warn "torch ${TORCH_VER} is newer than the pyproject pin (torch<=2.11.0); 'pip install -e .' may downgrade it. Pass --minimal and install deps yourself if you need to keep this torch."
        fi
    fi

    # `pip install -e .` covers pyproject's dependency list, but requirements.txt
    # carries several that pyproject omits (torchao, langdetect, zmq, jsonschema,
    # pymongo, modelscope). Install both so neither set is missing.
    #
    # One line has to be filtered out: requirements.txt still asks for
    # `sgl-kernel`, which was renamed to `sglang-kernel` upstream. The old name
    # is frozen at 0.3.21 while the new one is on 0.4.x, and BOTH install the
    # same `sgl_kernel` module -- so letting requirements.txt pull the stale one
    # would fight with the version installed by --quant sgl. Upstream's own
    # cu130 image already uses the new name.
    run_sh "${PIP} install -v -e '${REPO_ROOT}'"
    local req="${BUILD_DIR}/requirements.filtered.txt"
    run mkdir -p "${BUILD_DIR}"
    run_sh "grep -v -E '^[[:space:]]*sgl-kernel([[:space:]]|==|>=|<=|$)' '${REPO_ROOT}/requirements.txt' > '${req}'"
    run_sh "${PIP} install -r '${req}'"
    ok "core (pyproject + requirements.txt, minus the renamed sgl-kernel)"
}

# ----------------------------------------------------------------------------
# attention operators
# ----------------------------------------------------------------------------

install_fa4() {
    # Published wheel, not a source build -- this is the fast one (seconds, not
    # the 20-60 min a FlashAttention 2 source build costs). The kernels are
    # CuTe-DSL and JIT at first use.
    if [[ "${NVCC_CUDA}" == 13.* ]]; then
        log "FlashAttention 4 (CuTe DSL, cu13 extra)"
        run_sh "${PIP} install 'flash-attn-4[cu13]'"
    else
        log "FlashAttention 4 (CuTe DSL)"
        run_sh "${PIP} install flash-attn-4"
    fi
    ok "flash_attn4 (attn_type=flash_attn4 / spas_flash_attn4)"
}

install_sage2() {
    case "${ARCH}" in
        10.0|10.3)
            skip "sage_attn2: SageAttention 2 has no sm100/sm103 kernels; use fa4 (default) or sage3 on B200/B300"
            return 0 ;;
    esac
    log "SageAttention 2 (source)"
    fetch "https://github.com/ModelTC/SageAttention.git" "SageAttention" "--depth 1"
    # Only build what this branch targets; the upstream list also carries
    # consumer arches that just lengthen the build.
    local archs="${SAGE_CUDA_ARCHITECTURES:-9.0}"
    run_sh "cd '${BUILD_DIR}/SageAttention' && CUDA_ARCHITECTURES='${archs}' EXT_PARALLEL=4 NVCC_APPEND_FLAGS='--threads 8' MAX_JOBS=${JOBS} ${PIP} install --no-build-isolation -v -e ."
    ok "sage_attn2"
}

install_sage3() {
    case "${ARCH}" in
        10.0|10.3) : ;;
        "") skip "sage_attn3: no GPU detected; pass --arch 10.0 or 10.3 to force" ; return 0 ;;
        *)  skip "sage_attn3: needs Blackwell (B200/B300), found compute cap ${ARCH}" ; return 0 ;;
    esac
    log "SageAttention 3 / Blackwell FP4 attention (source)"
    fetch "https://github.com/ModelTC/SageAttention-1104.git" "SageAttention-1104" "--depth 1"
    run_sh "cd '${BUILD_DIR}/SageAttention-1104/sageattention3_blackwell' && MAX_JOBS=${JOBS} python setup.py install"
    ok "sage_attn3"
}

install_sparge() {
    case "${ARCH}" in
        10.0|10.3)
            skip "sparge_attn: upstream builds 8.0-9.0 and 12.0 only, no sm100/sm103"
            return 0 ;;
    esac
    log "SpargeAttn (source)"
    fetch "https://github.com/ModelTC/SpargeAttn.git" "SpargeAttn" "--depth 1"
    local archs="9.0"
    run_sh "cd '${BUILD_DIR}/SpargeAttn' && TORCH_CUDA_ARCH_LIST='${archs}' MAX_JOBS=${JOBS} ${PIP} install --no-build-isolation -v -e ."
    ok "sparge_attn"
}

install_magi() {
    log "MagiAttention (source)"
    fetch "https://github.com/SandAI-org/MagiAttention.git" "MagiAttention" "--recursive"
    local caps="90,100"   # the two in-scope families; upstream supports both
    run_sh "cd '${BUILD_DIR}/MagiAttention' && MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY='${caps}' MAX_JOBS=${JOBS} ${PIP} install --no-build-isolation -v -e ."
    ok "magi_attention"
}

install_attn() {
    local spec="$1"
    log "attention operators: ${spec}"
    local backends=()
    IFS=',' read -r -a backends <<< "${spec}"
    for backend in "${backends[@]}"; do
        case "${backend}" in
            fa4)    install_fa4 ;;
            sage2)  install_sage2 ;;
            sage3)  install_sage3 ;;
            sparge) install_sparge ;;
            magi)   install_magi ;;
            none)   skip "attention operators (explicitly disabled)" ;;
            *)      die "unknown attention backend '${backend}' (fa4|sage2|sage3|sparge|magi|none)" ;;
        esac
    done
}

# ----------------------------------------------------------------------------
# quantization operators
# ----------------------------------------------------------------------------

install_quant() {
    local spec="$1"
    log "quantization operators: ${spec}"
    local backends=()
    IFS=',' read -r -a backends <<< "${spec}"
    for backend in "${backends[@]}"; do
        case "${backend}" in
            sgl)
                # Package renamed: `sgl-kernel` (stuck at 0.3.21) -> `sglang-kernel`
                # (0.4.x). Both provide the `sgl_kernel` module. Pinned to the
                # version upstream's cu130 image validates against torch 2.11;
                # override with SGLANG_KERNEL_VERSION (latest is newer).
                #
                # This also backs LightX2V's default fused RMSNorm
                # (rms_norm_type "sgl-kernel"), not just the fp8/int8 GEMMs --
                # without it that path silently drops to pure torch.
                run_sh "${PIP} install 'sglang-kernel==${SGLANG_KERNEL_VERSION:-0.4.4}'"
                ok "sglang-kernel (fp8-sgl / int8-sgl + fused RMSNorm)" ;;
            vllm)
                # Pinned to upstream's cu130 image; bare `pip install vllm`
                # pulls a much newer release that has not been tried against
                # this torch. Override with VLLM_VERSION.
                run_sh "${PIP} install 'vllm==${VLLM_VERSION:-0.23.0}'"
                ok "vllm kernels (fp8-vllm / int8-vllm)" ;;
            torchao)
                run_sh "${PIP} install torchao"
                ok "torchao (fp8-torchao / int8-torchao)" ;;
            none)
                skip "quantization operators (explicitly disabled)" ;;
            *)
                die "unknown quant backend '${backend}' (sgl|vllm|torchao|none)" ;;
        esac
    done
}

# ----------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------

verify() {
    log "verifying installation"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "    (skipped in --dry-run)"
        return 0
    fi
    python3 - <<'PYEOF'
import importlib, sys

CORE = [("torch", None), ("safetensors", None), ("transformers", None), ("diffusers", None),
        ("einops", None), ("loguru", None), ("av", "video/audio muxing for LTX-2"),
        ("PIL", "image conditioning"), ("numpy", None)]
ATTN = [("flash_attn.cute", "attn_type=flash_attn4 / spas_flash_attn4"),
        ("flash_attn", "attn_type=flash_attn2 (not installed by this script)"),
        ("flash_attn_interface", "attn_type=flash_attn3 (not installed by this script)"),
        ("sageattention", "attn_type=sage_attn2"),
        ("sageattn3", "attn_type=sage_attn3"),
        ("spas_sage_attn", "attn_type=spas_sage_attn2"),
        ("magi_attention", "sequence-parallel MagiAttention")]
QUANT = [("sgl_kernel", "fp8-sgl / int8-sgl + fused RMSNorm"),
         ("vllm", "fp8-vllm / int8-vllm"),
         ("torchao", "fp8-torchao / int8-torchao"),
         ("gguf", "gguf-* schemes")]

def check(group, items, required):
    print(f"\n  {group}")
    missing = []
    for mod, note in items:
        try:
            importlib.import_module(mod)
            mark, state = "\033[32m ok \033[0m", ""
        except Exception:
            mark, state = ("\033[31mMISS\033[0m" if required else "\033[2m -- \033[0m"), ""
            missing.append(mod)
        suffix = f"   ({note})" if note else ""
        print(f"    [{mark}] {mod}{suffix}{state}")
    return missing

missing_core = check("core", CORE, True)
attn_missing = check("attention operators (optional)", ATTN, False)
check("quantization operators (optional)", QUANT, False)

print()
if missing_core:
    print(f"\033[31mFAIL\033[0m core dependencies missing: {missing_core}")
    sys.exit(1)

have_attn = [m for m, _ in ATTN if m not in attn_missing]
if not have_attn:
    print("\033[33mNOTE\033[0m no accelerated attention installed. The shipped LTX-2 configs")
    print("     use attn_type=sage_attn2, which will crash without sageattention.")
    print("     Set \"attn_type\": \"torch_sdpa\" in your config to run on pure torch.")
else:
    print(f"\033[32mOK\033[0m   accelerated attention available: {', '.join(have_attn)}")

try:
    import torch
    print(f"     torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"devices {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
except Exception as e:
    print(f"     torch check failed: {e}")
PYEOF
}

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

detect_env

if [[ "${VERIFY_ONLY}" == "1" ]]; then
    verify
    exit $?
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY RUN -- no commands will be executed"
fi

install_core

if [[ "${MINIMAL}" == "1" ]]; then
    skip "all attention operators (--minimal)"
    skip "all quantization operators (--minimal)"
else
    install_attn "${ATTN_SPEC:-$(default_attn_for_arch "${ARCH}")}"
    install_quant "${QUANT_SPEC:-$(default_quant_for_arch "${ARCH}")}"
fi

echo
log "summary"
for i in "${INSTALLED[@]}"; do printf '    \033[32m+\033[0m %s\n' "$i"; done
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    for s in "${SKIPPED[@]}"; do printf '    \033[33m-\033[0m %s\n' "$s"; done
fi

verify

echo
log "next steps"
cat <<EOF
    Smoke test (no model needed):
      python -c "import lightx2v; print('import ok')"

    Run LTX-2.3:
      python -m lightx2v.infer --model_cls ltx2 --task t2av \\
          --model_path /path/to/model --config_json configs/ltx2/ltx2_3.json \\
          --prompt "..." --save_result_path out.mp4

    If you skipped attention operators, add to your config:
      "attn_type": "torch_sdpa"
EOF
