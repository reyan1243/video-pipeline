#!/usr/bin/env bash
# Proves the captions-behind-subject composite works, using only ffmpeg.
#
# This is the exact layer stack the Remotion frontend will build, expressed as a
# filtergraph. If the output looks right here, the canvas implementation will
# look the same — so it is worth checking before writing any React.
#
#     source video                     <- background
#     caption text                     <- drawn over it
#     source video, masked by matte    <- the subject, back on top
#
# The subject's colour comes from a second copy of the SOURCE, never from the
# matte. That is the whole point: the matte carries only the silhouette.
#
# Usage:
#   scripts/verify_composite.sh INPUT.mp4 MATTE.mp4 [OUT.mp4] [SECONDS] [TEXT]

set -euo pipefail

INPUT=${1:?usage: verify_composite.sh INPUT.mp4 MATTE.mp4 [OUT.mp4] [SECONDS] [TEXT]}
MATTE=${2:?missing matte}
OUT=${3:-composite_proof.mp4}
SECONDS_TO_RENDER=${4:-10}
TEXT=${5:-CAPTIONS GO BEHIND}

for f in "$INPUT" "$MATTE"; do
  [ -f "$f" ] || { echo "not found: $f" >&2; exit 1; }
done

probe() { ffprobe -v error -select_streams v:0 -show_entries "stream=$2" -of csv=p=0 "$1"; }

W=$(probe "$INPUT" width); H=$(probe "$INPUT" height)
MW=$(probe "$MATTE" width); MH=$(probe "$MATTE" height)
echo "source ${W}x${H}   matte ${MW}x${MH}"
[ "$MW" = "$W" ] && [ "$MH" = "$H" ] || echo "  (matte is a different size — scaling it up, which is expected with --max-mask-height)"

# A font path that exists on both macOS and typical Linux containers.
FONT=""
for candidate in /System/Library/Fonts/Supplemental/Arial\ Bold.ttf \
                 /System/Library/Fonts/Helvetica.ttc \
                 /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; do
  [ -f "$candidate" ] && { FONT="fontfile=$candidate:"; break; }
done

FONTSIZE=$((W / 11))

ffmpeg -hide_banner -v warning -y \
  -t "$SECONDS_TO_RENDER" -i "$INPUT" \
  -t "$SECONDS_TO_RENDER" -i "$MATTE" \
  -filter_complex "
    [0:v]split=2[base][subject];
    [base]drawtext=${FONT}text='${TEXT}':fontsize=${FONTSIZE}:fontcolor=white:
          borderw=$((FONTSIZE/16)):bordercolor=black:
          x=(w-text_w)/2:y=h*0.42[captioned];
    [1:v]format=gray,scale=${W}:${H}[matte];
    [subject][matte]alphamerge[cutout];
    [captioned][cutout]overlay=0:0:format=auto[out]
  " \
  -map "[out]" -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p "$OUT"

echo
echo "wrote $OUT"
echo
echo "What to look for:"
echo "  1. Text passes BEHIND the subject, not over it."
echo "  2. The silhouette edge is soft, not stair-stepped."
echo "  3. Gaps — under an arm, between fingers — let the text show through."
echo "  4. The edge does not shimmer or crawl between frames."
echo "  5. No dark halo hugging the subject."
