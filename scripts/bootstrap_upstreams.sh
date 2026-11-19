#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_root="${project_root}/external"
patch_path="${project_root}/patches/diffgrm-position-loss.patch"
genrec_patch_path="${project_root}/patches/genrec-rqvae-amp.patch"

mkdir -p "${external_root}"

clone_or_checkout() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local target="${external_root}/${name}"

    if [[ ! -d "${target}/.git" ]]; then
        git clone "${url}" "${target}"
    fi
    git -C "${target}" checkout --detach "${commit}"
}

clone_or_checkout \
    "genrec" \
    "https://github.com/phonism/genrec.git" \
    "fb4bd1a1832df0312580a705d3440d1988856e31"

clone_or_checkout \
    "DiffGRM" \
    "https://github.com/liuzhao09/DiffGRM.git" \
    "ad7b971c7e525e9fea6fb8e362a5c49dccb2473c"

genrec_root="${external_root}/genrec"
if git -C "${genrec_root}" apply --check "${genrec_patch_path}" 2>/dev/null; then
    git -C "${genrec_root}" apply "${genrec_patch_path}"
    echo "Applied the CUDA-autocast Sinkhorn dtype fix to GenRec RQ-VAE."
elif git -C "${genrec_root}" apply --reverse --check "${genrec_patch_path}" 2>/dev/null; then
    echo "GenRec RQ-VAE Sinkhorn dtype fix is already applied."
else
    echo "GenRec RQ-VAE patch does not apply cleanly." >&2
    exit 1
fi

diff_root="${external_root}/DiffGRM"
if git -C "${diff_root}" apply --check "${patch_path}" 2>/dev/null; then
    git -C "${diff_root}" apply "${patch_path}"
    echo "Applied AR position-loss, collision-suffix, and strict-evaluation instrumentation to DiffGRM."
elif git -C "${diff_root}" apply --reverse --check "${patch_path}" 2>/dev/null; then
    echo "DiffGRM AR experiment instrumentation is already applied."
else
    echo "DiffGRM patch does not apply cleanly. Check the pinned commit and local changes." >&2
    exit 1
fi
