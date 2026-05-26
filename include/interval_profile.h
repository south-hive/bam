/*
 * Per-thread interval sampler for the rand_read_bam vs rand_read_split
 * A/B study.
 *
 * Sampling model: we record absolute device clock64() timestamps at
 * each phase boundary for the first PROF_K_SAMPLES requests issued by
 * each thread. The host derives per-phase, per-request intervals by
 * subtracting consecutive timestamps (the very first interval uses
 * `birth`, captured when the thread enters the issue loop).
 *
 * This is more faithful than the previous "sum cycles per phase"
 * model: it preserves the actual back-to-back timeline of K requests
 * per thread, so a downstream visualizer can render the real
 * 32-command sequence rather than replicate a single averaged cmd.
 *
 * Phase indices into stamps[r][...]:
 *   0  cid_acquire        (end of)
 *   1  cmd_build          (end of)
 *   2  sq_ticket_wait     (end of)
 *   3  sq_cmd_copy        (end of)
 *   4  sq_tail_advance    (end of)
 *   5  cq_poll_scan       (end of)
 *   6  cq_head_advance    (end of)   -- this is also the request boundary
 */
#ifndef __INTERVAL_PROFILE_H__
#define __INTERVAL_PROFILE_H__

#ifndef __device__
#define __device__
#endif

#include <stdint.h>

#define PROF_K_SAMPLES 32
#define PROF_N_PHASES  7

/* Phase indices (must match the comment block above). */
#define PROF_PH_CID_ACQUIRE     0
#define PROF_PH_CMD_BUILD       1
#define PROF_PH_SQ_TICKET_WAIT  2
#define PROF_PH_SQ_CMD_COPY     3
#define PROF_PH_SQ_TAIL_ADVANCE 4
#define PROF_PH_CQ_POLL_SCAN    5
#define PROF_PH_CQ_HEAD_ADVANCE 6

struct interval_record_t {
    uint64_t birth;
    uint32_t cur_req;
    uint32_t _pad0;
    uint64_t stamps[PROF_K_SAMPLES][PROF_N_PHASES];
};

/* Capture the per-thread birth timestamp once, before the issue loop. */
#define PROF_INIT_BIRTH(rec) do {                                       \
    if (rec) {                                                          \
        (rec)->birth = clock64();                                       \
        (rec)->cur_req = 0;                                             \
    }                                                                   \
} while (0)

/*
 * Stamp the absolute clock at the end of a phase, into the current
 * request slot. No-op if the thread has already filled its K slots.
 */
#define PROF_STAMP(rec, phase) do {                                     \
    if ((rec) && (rec)->cur_req < PROF_K_SAMPLES) {                     \
        (rec)->stamps[(rec)->cur_req][(phase)] = clock64();             \
    }                                                                   \
} while (0)

/* Advance to the next request slot. Saturates at PROF_K_SAMPLES. */
#define PROF_NEXT_REQ(rec) do {                                         \
    if ((rec) && (rec)->cur_req < PROF_K_SAMPLES) {                     \
        (rec)->cur_req++;                                               \
    }                                                                   \
} while (0)

#endif // __INTERVAL_PROFILE_H__
