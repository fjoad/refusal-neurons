#!/bin/bash
# Fonts for omni.py's pictures in other scripts (Noto, SIL Open Font License), into assets/fonts/ (gitignored;
# about 18 MB). Run once on each machine: bash scripts/get_fonts.sh
set -e
mkdir -p "$(dirname "$0")/../assets/fonts" && cd "$(dirname "$0")/../assets/fonts"
B=https://github.com/notofonts/notofonts.github.io/raw/main/fonts
for f in NotoSans NotoSansArabic NotoSansThai NotoSansBengali; do
  [ -s $f-Regular.ttf ] || curl -sL --retry 3 -o $f-Regular.ttf "$B/$f/hinted/ttf/$f-Regular.ttf"
done
[ -s NotoSansCJKsc-Regular.otf ] || curl -sL --retry 3 -o NotoSansCJKsc-Regular.otf \
  https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
ls -la
