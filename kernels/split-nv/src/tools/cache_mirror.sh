#!/bin/bash
# Box prefix cache lives in /dev/shm (lost on reboot). `sync` mirrors it to local NVMe; `restore` copies the mirror
# back when /dev/shm is empty (i.e. after a reboot), before the engine starts. Files are immutable once written and an
# entry exists only once its idx json does, so a mirror taken mid-write is safe: the loader drops incomplete entries.
set -u
SRC=${SPLIT_NV_CACHE_DIR:-/dev/shm/split-nv/cache}
MIRROR=${SPLIT_NV_CACHE_MIRROR:-/mnt/nvme-1/split-nv-cache}
MIN_FREE_GB=${SPLIT_NV_MIRROR_MIN_FREE_GB:-60}
exec 9>/tmp/split-nv-cache-mirror.lock; flock 9
nfiles() { find "$1" -type f 2>/dev/null | head -1 | wc -l; }
case "${1:-}" in
  sync)
    # never mirror an empty cache over a full mirror (fresh boot before restore, or shm wiped)
    [ "$(nfiles "$SRC")" = 1 ] || { echo "cache empty; not syncing"; exit 0; }
    mkdir -p "$MIRROR"
    free=$(df -BG --output=avail "$MIRROR" | tail -1 | tr -dc 0-9)
    [ "$free" -ge "$MIN_FREE_GB" ] || { echo "mirror disk has ${free} GB free (< ${MIN_FREE_GB}); skipping"; exit 0; }
    nice -n 19 ionice -c3 rsync -a --delete --exclude '*.tmp*' "$SRC"/ "$MIRROR"/ ;;
  restore)
    [ "$(nfiles "$SRC")" = 0 ] || { echo "cache present; no restore"; exit 0; }
    [ "$(nfiles "$MIRROR")" = 1 ] || { echo "no mirror"; exit 0; }
    mkdir -p "$SRC"; t=$(date +%s)
    rsync -a "$MIRROR"/ "$SRC"/ && echo "restored $(du -sh "$SRC" | cut -f1) in $(( $(date +%s) - t )) s" ;;
  *) echo "usage: $0 sync|restore"; exit 2 ;;
esac
