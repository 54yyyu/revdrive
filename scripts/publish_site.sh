#!/bin/sh
# Publish site/ to the gh-pages branch, which GitHub Pages serves.
# The branch holds only the latest build (videos included), so main stays small:
# each publish replaces it. Build first: python scripts/build_site.py
set -eu
cd "$(dirname "$0")/.."
[ -f site/index.html ] || { echo "no site/ yet: python scripts/build_site.py"; exit 1; }
remote=$(git remote get-url origin)
head=$(git rev-parse --short HEAD)
cd site
rm -rf .git
git init -q -b gh-pages
git add -A
git commit -qm "Site built from $head"
git push -qf "$remote" gh-pages
rm -rf .git
echo "published site built from $head to gh-pages"
