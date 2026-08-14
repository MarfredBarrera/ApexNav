#!/usr/bin/env bash
#
# Host-side, run once. Populates the data tree that gets bind-mounted into the
# container at /workspace/ApexNav/data.
#
#   ./scripts/setup_data.sh
#
# Everything is idempotent: already-present files are skipped, so re-running after
# an interrupted download is safe.
#
# Override with environment variables:
#   APEXNAV_DATA  where the data tree lives         (default ~/nas/apexnav_data)
#   HM3D_SRC      existing HM3D-v0.2 scenes to copy (default the DC-ObjectNav copy)
#   SKIP_MP3D=1   skip the MP3D episode download

set -euo pipefail

APEXNAV_DATA="${APEXNAV_DATA:-$HOME/nas/apexnav_data}"
HM3D_SRC="${HM3D_SRC:-$HOME/codes/DC-ObjectNav/DCON/versioned_data/hm3d-0.2/hm3d}"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
skip() { printf '    \033[2m(present, skipping) %s\033[0m\n' "$*"; }

mkdir -p "${APEXNAV_DATA}"/{scene_datasets,datasets/objectnav/hm3d,datasets/objectnav/mp3d,ollama}

# ---------------------------------------------------------------------------
# 1. HM3D-v0.2 scenes
# ---------------------------------------------------------------------------
# Copied rather than downloaded: the local tree is the same hm3d-val-habitat-v0.2
# content (.basis.glb + .basis.navmesh + .semantic.glb + .semantic.txt per scene),
# which skips 5.3 GB of transfer and the Matterport permission step.
log "HM3D-v0.2 scenes -> ${APEXNAV_DATA}/scene_datasets/hm3d"
if [ -d "${APEXNAV_DATA}/scene_datasets/hm3d/val" ] \
   && [ "$(ls -1 "${APEXNAV_DATA}/scene_datasets/hm3d/val" 2>/dev/null | wc -l)" -ge 100 ]; then
    skip "$(ls -1 "${APEXNAV_DATA}/scene_datasets/hm3d/val" | wc -l) scene directories"
else
    if [ ! -d "${HM3D_SRC}/val" ]; then
        echo "error: no HM3D scenes at ${HM3D_SRC}/val" >&2
        echo "       set HM3D_SRC, or download hm3d-val-habitat-v0.2.tar from Matterport." >&2
        exit 1
    fi
    mkdir -p "${APEXNAV_DATA}/scene_datasets/hm3d"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --info=progress2 "${HM3D_SRC}/" "${APEXNAV_DATA}/scene_datasets/hm3d/"
    else
        cp -a "${HM3D_SRC}/." "${APEXNAV_DATA}/scene_datasets/hm3d/"
    fi
fi

# ApexNav's configs reference both names; the README makes v0.2 scenes serve both.
if [ ! -e "${APEXNAV_DATA}/scene_datasets/hm3d_v0.2" ]; then
    ln -s hm3d "${APEXNAV_DATA}/scene_datasets/hm3d_v0.2"
    log "linked scene_datasets/hm3d_v0.2 -> hm3d"
fi

# ---------------------------------------------------------------------------
# 2. ObjectNav episode datasets
# ---------------------------------------------------------------------------
fetch_zip() {  # url  dest_marker_dir  tmp_zip  post-extract shell snippet
    local url="$1" marker="$2" tmp="$3"
    if [ -d "${marker}" ]; then skip "${marker}"; return 1; fi
    log "downloading $(basename "${url}")"
    wget -q --show-progress -O "${tmp}" "${url}"
    return 0
}

cd "${APEXNAV_DATA}"

if fetch_zip \
      "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip" \
      "datasets/objectnav/hm3d/v1" /tmp/objectnav_hm3d_v1.zip; then
    unzip -q /tmp/objectnav_hm3d_v1.zip -d datasets/objectnav/hm3d
    mv datasets/objectnav/hm3d/objectnav_hm3d_v1 datasets/objectnav/hm3d/v1
    rm /tmp/objectnav_hm3d_v1.zip
fi

if fetch_zip \
      "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip" \
      "datasets/objectnav/hm3d/v2" /tmp/objectnav_hm3d_v2.zip; then
    unzip -q /tmp/objectnav_hm3d_v2.zip -d datasets/objectnav/hm3d
    mv datasets/objectnav/hm3d/objectnav_hm3d_v2 datasets/objectnav/hm3d/v2
    rm /tmp/objectnav_hm3d_v2.zip
fi

# Not optional even for HM3D runs: habitat_evaluation.py reads
# data/datasets/objectnav/mp3d/v1/val/val.json.gz for its category-name mapping.
# The 'm3d' in the path is not a typo - it is the URL that actually serves; the
# 'mp3d' spelling returns 403.
if [ -z "${SKIP_MP3D:-}" ]; then
    if fetch_zip \
          "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip" \
          "datasets/objectnav/mp3d/v1" /tmp/objectnav_mp3d_v1.zip; then
        unzip -q /tmp/objectnav_mp3d_v1.zip -d datasets/objectnav/mp3d/v1
        rm /tmp/objectnav_mp3d_v1.zip
    fi
fi

# ---------------------------------------------------------------------------
# 3. Model weights
# ---------------------------------------------------------------------------
fetch_weight() {  # url  filename
    local url="$1" out="$2"
    if [ -s "${out}" ]; then skip "${out}"; return; fi
    log "downloading ${out}"
    wget -q --show-progress -O "${out}.part" "${url}"
    mv "${out}.part" "${out}"
}

fetch_weight "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
             groundingdino_swint_ogc.pth
fetch_weight "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt" \
             yolov7-e6e.pt
fetch_weight "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt" \
             mobile_sam.pt

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
log "data tree at ${APEXNAV_DATA}"
missing=0
for p in scene_datasets/hm3d/val scene_datasets/hm3d_v0.2 \
         datasets/objectnav/hm3d/v1/val datasets/objectnav/hm3d/v2/val \
         datasets/objectnav/mp3d/v1/val \
         groundingdino_swint_ogc.pth mobile_sam.pt yolov7-e6e.pt; do
    if [ -e "${APEXNAV_DATA}/${p}" ]; then
        printf '  \033[32mok\033[0m    %s\n' "${p}"
    else
        printf '  \033[31mMISSING\033[0m %s\n' "${p}"
        missing=1
    fi
done
printf '\n  total: %s\n' "$(du -sh "${APEXNAV_DATA}" 2>/dev/null | cut -f1)"

if [ "${missing}" -ne 0 ]; then
    echo
    echo "Some items are missing - re-run this script, or see docs/DOCKER_JAZZY_FOXGLOVE.md." >&2
    exit 1
fi

cat <<'EOF'

Next:
  cd docker && docker compose up -d
  docker compose exec apexnav ./scripts/build_ws.sh
EOF
