#!/bin/bash
# Box prefix cache lives in /dev/shm (lost on reboot). `sync` mirrors it to local NVMe; `restore` copies the mirror
# back when /dev/shm is empty (i.e. after a reboot), before the engine starts. Files are immutable once written and an
# entry exists only once its idx json does, so a mirror taken mid-write is safe: the loader drops incomplete entries.
#
# The mirror is the only persistent copy and sync runs rsync --delete, so a partial source must never be synced.
# Markers in SRC (tmpfs: gone after every reboot):
#   restore  removes .restore-done, writes .restore-running, copies, and only after rsync succeeded writes
#            .restore-done and removes .restore-running. A restore that did not finish is resumed by the next
#            restore (rsync is incremental) instead of being taken for a present cache.
#   sync     runs only when .restore-done exists AND the source holds >= MIN_PCT (90) % of the mirror's files (or the
#            mirror is empty). No marker at all means this boot did not restore (the engine built the cache, or it
#            predates the markers): sync adopts the source (writes .restore-done) when the count check passes.
#            .restore-running without .restore-done (a partial restore) is refused.
# Refusals exit 75 (EX_TEMPFAIL, shows in `systemctl --user --failed`) and copy nothing. After an intentional
# /v1/cache/clear the source is small on purpose: empty the mirror first (rm -rf "$MIRROR"/*), or run one sync with
# SPLIT_NV_MIRROR_MIN_PCT=0.
set -u
SRC=${SPLIT_NV_CACHE_DIR:-/dev/shm/split-nv/cache}
MIRROR=${SPLIT_NV_CACHE_MIRROR:-/mnt/nvme-1/split-nv-cache}
MIN_FREE_GB=${SPLIT_NV_MIRROR_MIN_FREE_GB:-60}
MIN_PCT=${SPLIT_NV_MIRROR_MIN_PCT:-90}
DONE="$SRC/.restore-done"
RUNNING="$SRC/.restore-running"
exec 9>"${SPLIT_NV_MIRROR_LOCK:-/tmp/split-nv-cache-mirror.lock}"; flock 9
nfiles() { find "$1" -type f ! -name '.restore-*' 2>/dev/null | head -1 | wc -l; }
count() { find "$1" -type f ! -name '.restore-*' ! -name '*.tmp*' 2>/dev/null | wc -l; }
case "${1:-}" in
  sync)
    # never mirror an empty cache over a full mirror (fresh boot before restore, or shm wiped)
    [ "$(nfiles "$SRC")" = 1 ] || { echo "cache empty; not syncing"; exit 0; }
    if [ ! -e "$DONE" ] && [ -e "$RUNNING" ]; then
      echo "the restore of this boot did not finish ($RUNNING); not syncing (finish it: $0 restore)"; exit 75
    fi
    mkdir -p "$MIRROR"
    free=$(df -BG --output=avail "$MIRROR" | tail -1 | tr -dc 0-9)
    [ "$free" -ge "$MIN_FREE_GB" ] || { echo "mirror disk has ${free} GB free (< ${MIN_FREE_GB}); skipping"; exit 0; }
    src=$(count "$SRC"); mir=$(count "$MIRROR")
    if [ "$mir" -gt 0 ] && [ $((src * 100)) -lt $((mir * MIN_PCT)) ]; then
      echo "source has $src files, the mirror $mir (< ${MIN_PCT}%): not syncing (--delete would drop mirror files)"; exit 75
    fi
    if [ ! -e "$DONE" ]; then
      touch "$DONE" && echo "no restore marker (cache not restored this boot): adopted the source ($src files, mirror $mir)"
    fi
    nice -n 19 ionice -c3 rsync -a --delete --exclude '*.tmp*' --exclude '/.restore-*' "$SRC"/ "$MIRROR"/ ;;
  restore)
    if [ -e "$RUNNING" ]; then
      echo "resuming a restore that did not finish"
    else
      [ "$(nfiles "$SRC")" = 0 ] || { echo "cache present; no restore"; exit 0; }
    fi
    [ "$(nfiles "$MIRROR")" = 1 ] || { echo "no mirror"; exit 0; }
    mkdir -p "$SRC" && rm -f "$DONE" && touch "$RUNNING" || { echo "cannot write the restore marker in $SRC"; exit 1; }
    t=$(date +%s)
    rsync -a --exclude '*.tmp*' --exclude '/.restore-*' "$MIRROR"/ "$SRC"/
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "restore failed (rsync exit $rc): $RUNNING left, sync refuses until a restore completes"; exit $rc
    fi
    touch "$DONE" && rm -f "$RUNNING"
    echo "restored $(du -sh "$SRC" | cut -f1) in $(( $(date +%s) - t )) s" ;;
  *) echo "usage: $0 sync|restore"; exit 2 ;;
esac
