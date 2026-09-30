#!/bin/bash
# Shell test of tools/cache_mirror.sh on temp dirs (never the real cache or mirror): restore markers, the partial-
# restore guard, the file-count guard, first-boot adoption, and that markers never reach the mirror.
#   usage: tools/test_cache_mirror.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/cache-mirror-test.XXXXXX")
trap 'rm -rf "$T"' EXIT
export SPLIT_NV_CACHE_DIR="$T/shm/cache" SPLIT_NV_CACHE_MIRROR="$T/nvme/mirror" SPLIT_NV_MIRROR_LOCK="$T/lock" \
  SPLIT_NV_MIRROR_MIN_FREE_GB=0
S=$SPLIT_NV_CACHE_DIR M=$SPLIT_NV_CACHE_MIRROR
fails=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
run() { bash "$HERE/cache_mirror.sh" "$@" >"$T/out" 2>&1; echo $? >"$T/rc"; }
rc() { cat "$T/rc"; }
listing() { (cd "$1" 2>/dev/null && find . -type f ! -name '.restore-*' | sort); }
same() { [ "$(listing "$S")" = "$(listing "$M")" ]; }
populate() {  # $1 dir, $2 entries: 5 entry files + 3 block files each, like cache/<numerics>/
  mkdir -p "$1/og-s4.4"
  for i in $(seq 1 "$2"); do
    for f in idx-e$i.json tok-e$i.npy rows-tail-e$i ent-e$i.r0 ent-e$i.r1 blk-b$i.r0 blk-b$i.r1 rows-b$i; do
      echo "$f" >"$1/og-s4.4/$f"
    done
  done
}
reset() { rm -rf "$T/shm" "$T/nvme"; mkdir -p "$S" "$M"; }
evict() {  # $1 dir, $2 entry number: its 5 entry files and its block's 3 files
  local i=$2
  rm -f "$1"/og-s4.4/{idx-e$i.json,tok-e$i.npy,rows-tail-e$i,ent-e$i.r0,ent-e$i.r1,blk-b$i.r0,blk-b$i.r1,rows-b$i}
}

# 1. restore after a reboot: empty SRC, full mirror
reset; populate "$M" 100
run restore
check "restore: copies everything, exit 0" '[ "$(rc)" = 0 ] && same'
check "restore: .restore-done written, .restore-running removed" '[ -e "$S/.restore-done" ] && [ ! -e "$S/.restore-running" ]'
# 2. sync after a restore propagates evictions and new files
evict "$S" 1; evict "$S" 2
echo new >"$S/og-s4.4/idx-e999.json"
run sync
check "sync after restore: evictions + new files mirrored" '[ "$(rc)" = 0 ] && same && [ ! -e "$M/og-s4.4/idx-e1.json" ]'
check "markers never reach the mirror" '[ -z "$(find "$M" -name ".restore-*")" ]'
# 3. restore interrupted (rsync dies after part of the copy): sync must refuse, the mirror stays whole
reset; populate "$M" 100
mkdir -p "$T/bin"
cat >"$T/bin/rsync" <<'EOF'
#!/bin/bash
# the copy stops part way: every idx json (and everything after it) missing, rsync exits 23 (partial transfer)
/usr/bin/rsync --exclude 'idx-*' "$@"; exit 23
EOF
chmod +x "$T/bin/rsync"
PATH="$T/bin:$PATH" run restore
check "interrupted restore: non-zero exit, .restore-running left, no .restore-done" \
  '[ "$(rc)" = 23 ] && [ -e "$S/.restore-running" ] && [ ! -e "$S/.restore-done" ]'
before=$(listing "$M")
rm "$S"/og-s4.4/blk-b5*  # the loader or anything else touching the partial cache
run sync
check "sync after an interrupted restore: refused (75), mirror untouched" '[ "$(rc)" = 75 ] && [ "$(listing "$M")" = "$before" ]'
# 4. the next restore resumes and completes; then sync works
run restore
check "restore resumes a partial restore to completion" '[ "$(rc)" = 0 ] && same && [ -e "$S/.restore-done" ] && [ ! -e "$S/.restore-running" ]'
run sync
check "sync after the completed restore" '[ "$(rc)" = 0 ] && same'
# 5. first boot after this change: cache present from before, no markers, counts sane -> adopted
reset; populate "$M" 100; populate "$S" 100
evict "$S" 3
run sync
check "no markers, counts sane: source adopted (.restore-done), evictions mirrored" \
  '[ "$(rc)" = 0 ] && [ -e "$S/.restore-done" ] && same && grep -q adopted "$T/out"'
# 6. no markers and half the files: refused, not adopted
reset; populate "$M" 100; populate "$S" 50
before=$(listing "$M")
run sync
check "no markers, source at 50% of the mirror: refused (75), not adopted, mirror untouched" \
  '[ "$(rc)" = 75 ] && [ ! -e "$S/.restore-done" ] && [ "$(listing "$M")" = "$before" ]'
# 7. marker present but the source shrank below 90%
reset; populate "$M" 100; populate "$S" 85; touch "$S/.restore-done"
before=$(listing "$M")
run sync
check "marker present, source at 85%: refused (75), mirror untouched" '[ "$(rc)" = 75 ] && [ "$(listing "$M")" = "$before" ]'
populate "$S" 92
run sync
check "marker present, source at 92%: synced" '[ "$(rc)" = 0 ] && same'
# 8. empty source (markers only) never syncs
reset; populate "$M" 10; touch "$S/.restore-done"
before=$(listing "$M")
run sync
check "empty source: no sync, exit 0, mirror untouched" '[ "$(rc)" = 0 ] && [ "$(listing "$M")" = "$before" ] && grep -q "cache empty" "$T/out"'
# 9. no mirror yet (first boot ever / mirror wiped): the engine-built cache is adopted and mirrored
reset; populate "$S" 20
run restore
check "restore with an empty mirror: no-op, no markers" '[ "$(rc)" = 0 ] && [ -z "$(find "$S" -name ".restore-*")" ]'
run sync
check "sync into an empty mirror: adopted and mirrored" '[ "$(rc)" = 0 ] && [ -e "$S/.restore-done" ] && same'
# 10. restore with a present (restored) cache is a no-op
run restore
check "restore with the cache present: no-op" '[ "$(rc)" = 0 ] && grep -q "cache present" "$T/out"'
# 11. intentional clear: refused by default, SPLIT_NV_MIRROR_MIN_PCT=0 syncs once
reset; populate "$M" 100; populate "$S" 2; touch "$S/.restore-done"
run sync
check "intentional clear: refused by default" '[ "$(rc)" = 75 ]'
SPLIT_NV_MIRROR_MIN_PCT=0 run sync
check "intentional clear: SPLIT_NV_MIRROR_MIN_PCT=0 syncs" '[ "$(rc)" = 0 ] && same'

[ $fails = 0 ] && echo "ALL PASS" || echo "$fails FAILED"
exit $fails
