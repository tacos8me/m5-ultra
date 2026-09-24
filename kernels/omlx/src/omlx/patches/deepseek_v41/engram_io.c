// SPDX-License-Identifier: Apache-2.0
// Parallel positional row gather for SSD-offloaded DeepSeek V4.1 Engram tables.
//
// Random 256-byte rows are IOPS-bound: one blocking pread per row per thread
// keeps the SSD queue deep without Python (or the GIL) on the per-row path.
// With align > 0 each row is read as whole aligned blocks through a bounce
// buffer, which a F_NOCACHE descriptor serves as direct I/O.
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int fd;
    int64_t base;
    int64_t row_bytes;
    const int64_t *rows;
    int64_t n;
    uint8_t *out;
    int64_t align;
    int64_t batch;
    _Atomic int64_t next;
    _Atomic int err;
} job_t;

// Reads len bytes, or stops at end of file once at least need bytes arrived
// (an aligned block may extend past the end of the table file).
static int read_full(int fd, uint8_t *dst, int64_t len, int64_t need, int64_t off) {
    int64_t done = 0;
    while (done < len) {
        ssize_t got = pread(fd, dst + done, (size_t)(len - done), (off_t)(off + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            return errno ? errno : EIO;
        }
        if (got == 0) return done >= need ? 0 : EIO;
        done += got;
    }
    return 0;
}

static void *worker(void *arg) {
    job_t *job = (job_t *)arg;
    uint8_t *block = NULL;
    int64_t cap = 0;
    int rc = 0;
    for (;;) {
        int64_t first = atomic_fetch_add(&job->next, job->batch);
        if (first >= job->n || atomic_load(&job->err)) break;
        int64_t last = first + job->batch < job->n ? first + job->batch : job->n;
        for (int64_t i = first; i < last && !rc; i++) {
            int64_t off = job->base + job->rows[i] * job->row_bytes;
            uint8_t *dst = job->out + i * job->row_bytes;
            if (!job->align) {
                rc = read_full(job->fd, dst, job->row_bytes, job->row_bytes, off);
                continue;
            }
            int64_t lo = off / job->align * job->align;
            int64_t hi = (off + job->row_bytes + job->align - 1) / job->align * job->align;
            if (hi - lo > cap) {
                free(block);
                block = NULL;
                cap = hi - lo;
                if (posix_memalign((void **)&block, (size_t)job->align, (size_t)cap)) {
                    block = NULL;
                    rc = ENOMEM;
                    break;
                }
            }
            rc = read_full(job->fd, block, hi - lo, off - lo + job->row_bytes, lo);
            if (!rc) memcpy(dst, block + (off - lo), (size_t)job->row_bytes);
        }
        if (rc) {
            atomic_store(&job->err, rc);
            break;
        }
    }
    free(block);
    return NULL;
}

// Returns 0 or an errno value. rows are table row indices (bounds-checked by
// the caller); out receives n * row_bytes bytes in the order of rows.
int ds41_engram_gather(int fd, int64_t base, int64_t row_bytes, const int64_t *rows,
                       int64_t n, uint8_t *out, int threads, int64_t align) {
    if (n <= 0) return 0;
    if (threads < 1) threads = 1;
    if (threads > 256) threads = 256;
    if (threads > n) threads = (int)n;
    int64_t batch = n / ((int64_t)threads * 8);
    if (batch < 1) batch = 1;
    if (batch > 32) batch = 32;
    job_t job;
    memset(&job, 0, sizeof(job));
    job.fd = fd;
    job.base = base;
    job.row_bytes = row_bytes;
    job.rows = rows;
    job.n = n;
    job.out = out;
    job.align = align;
    job.batch = batch;
    atomic_init(&job.next, 0);
    atomic_init(&job.err, 0);
    pthread_t tids[256];
    int started = 0;
    for (; started < threads - 1; started++) {
        if (pthread_create(&tids[started], NULL, worker, &job)) break;
    }
    worker(&job);
    for (int i = 0; i < started; i++) pthread_join(tids[i], NULL);
    return atomic_load(&job.err);
}
