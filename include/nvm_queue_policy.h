/*
 * Indiscriminate completion-queue polling for the split-warp model.
 *
 * Provides a coexisting CQ-management protocol alongside the default
 * `cq_poll` / `cq_dequeue` in `nvm_parallel_queue.h`. The same
 * claim-ticket + representative phase-probe protocol is used, but the
 * ticket counter, election lock, and observed device tail live in an
 * *external* per-CQ state object (`nvm_indiscriminate_cq_state_t`)
 * rather than in `nvm_queue_t`. This leaves `cq->tail`,
 * `cq->cq_claim`, and `cq->cq_poll_lock` untouched, so the existing
 * BaM CQ behavior is preserved bit-for-bit and callers can mix the
 * two policies per queue without runtime branches on the hot path.
 *
 * Intended use: a split-warp kernel in which SQ enqueue and CQ drain
 * run on disjoint warps. cid pool entries are then acquired by SQ
 * warps (`get_cid`) and released by CQ warps (`put_cid`) — a
 * cross-warp share that is safe because `get_cid` / `put_cid` already
 * use device-wide atomics on `qp->sq.cid_pool`. Use `submit_cmd` from
 * SQ warps and `complete_cmd` from CQ warps; both are blocking and
 * the harness is responsible for issuing matching counts.
 */
#ifndef __NVM_QUEUE_POLICY_H__
#define __NVM_QUEUE_POLICY_H__

#ifndef __device__
#define __device__
#endif

#include <stdint.h>
#include <simt/atomic>

#include "nvm_types.h"
#include "nvm_cmd.h"
#include "nvm_parallel_queue.h"
#include "page_cache.h"
#include "queue.h"

/*
 * Per-CQ state for the indiscriminate-polling policy.
 *
 *   - claim         : monotonic completion claim-ticket counter
 *                     (`fetch_add(1)` per polling thread).
 *   - poll_lock     : representative phase-probe election lock
 *                     (1 = held, 0 = free). Same `LOCKED`/`UNLOCKED`
 *                     encoding as `tail_lock` in `nvm_parallel_queue.h`.
 *   - observed_tail : representative-published device-produced next
 *                     CQ slot. Replaces `cq->tail` for this policy
 *                     so the default meaning of `cq->tail` is preserved.
 *
 * Layout intentionally mirrors `nvm_queue_t` so each atomic sits on
 * its own 32-byte slot, avoiding false sharing between the three
 * counters on the hot polling path.
 */
struct nvm_indiscriminate_cq_state_t {
    simt::atomic<uint32_t, simt::thread_scope_device> claim;
    uint8_t pad0[28];
    simt::atomic<uint32_t, simt::thread_scope_device> poll_lock;
    uint8_t pad1[28];
    simt::atomic<uint32_t, simt::thread_scope_device> observed_tail;
    uint8_t pad2[28];
};

/*
 * `cq_poll_indiscriminate`
 *
 * Same claim-ticket + representative phase-probe protocol as
 * `cq_poll` in `nvm_parallel_queue.h`, but the ticket counter,
 * election lock, and observed device tail live in `state` rather
 * than in `nvm_queue_t`. `cq->tail` is never read or written here.
 *
 * Returns the CQ ring slot index (`ticket & qs_minus_1`) of the
 * arbitrary completion this caller has claimed. The caller must read
 * the completed CID from that slot before `cq_dequeue_indiscriminate`
 * advances head (subsequent phase rounds may overwrite the slot).
 */
inline __device__
uint32_t cq_poll_indiscriminate(nvm_queue_t* cq,
                                nvm_indiscriminate_cq_state_t* state,
                                uint32_t* claimed_pos = nullptr,
                                uint32_t* wait_from = nullptr) {
    uint32_t ticket = state->claim.fetch_add(1, simt::memory_order_relaxed);
    unsigned int ns = 8;
    do {
        bool acquired = false;
        if (state->poll_lock.load(simt::memory_order_relaxed) == UNLOCKED) {
            acquired = (state->poll_lock.fetch_or(LOCKED, simt::memory_order_acquire) == UNLOCKED);
        }
        if (acquired) {
            uint32_t t = state->observed_tail.load(simt::memory_order_relaxed);
            uint32_t expected = ((~(t >> cq->qs_log2)) & 1);
            while (true) {
                uint32_t loc = t & cq->qs_minus_1;
                // Plain read for speed; only the phase bit is consumed.
                uint32_t cpl = ((nvm_cpl_t*)cq->vaddr)[loc].dword[3];
                uint32_t phase = (cpl & 0x00010000) >> 16;
                if (phase != expected) {
                    break;
                }
                t++;
                expected = ((~(t >> cq->qs_log2)) & 1);
            }
            state->observed_tail.store(t, simt::memory_order_release);
            state->poll_lock.store(UNLOCKED, simt::memory_order_release);
        }

        uint32_t obs = state->observed_tail.load(simt::memory_order_acquire);
        if ((int32_t)(obs - ticket) > 0) {
            break;
        }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    } while (true);

    if (claimed_pos) *claimed_pos = ticket;
    if (wait_from) *wait_from = ticket;
    return ticket & cq->qs_minus_1;
}

/*
 * `cq_dequeue_indiscriminate`
 *
 * Provided for symmetry with `cq_poll_indiscriminate`. `cq_dequeue`
 * already takes `claimed_pos` / `wait_from` and never touches
 * `cq->tail`, so we forward to it. Kept as a named helper so the
 * indiscriminate path stays grep-able and any future divergence
 * lands here without disturbing the default call sites.
 */
inline __device__
void cq_dequeue_indiscriminate(nvm_queue_t* cq,
                               uint16_t cq_slot,
                               nvm_queue_t* sq,
                               nvm_indiscriminate_cq_state_t* /*state*/,
                               uint32_t claimed_pos,
                               uint32_t wait_from) {
    cq_dequeue(cq, cq_slot, sq, claimed_pos, wait_from);
}

/*
 * `submit_cmd`
 *
 * SQ-only warp helper. Builds one NVMe READ command and enqueues it
 * to `qp->sq`. The CQ is never touched.
 */
inline __device__
void submit_cmd(page_cache_d_t* pc,
                QueuePair* qp,
                uint64_t lba,
                uint64_t n_blocks,
                unsigned long long pc_entry) {
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    (void) sq_pos;
}

/*
 * `complete_cmd`
 *
 * CQ-only warp helper. Claims one arbitrary completion via the
 * indiscriminate-polling protocol, recovers the completed cid, and
 * returns the cid to the pool. Does not enqueue to the SQ.
 */
inline __device__
void complete_cmd(QueuePair* qp,
                  nvm_indiscriminate_cq_state_t* cq_state) {
    uint32_t claimed_pos = 0;
    uint32_t wait_from = 0;
    uint32_t cq_slot = cq_poll_indiscriminate(&qp->cq, cq_state,
                                              &claimed_pos, &wait_from);

    // Read cpl-cid before dequeue; dequeue may move head past this
    // slot and a later phase round can overwrite it.
    uint16_t cpl_cid =
        ((nvm_cpl_t*)qp->cq.vaddr)[cq_slot].dword[3] & 0xffff;

    cq_dequeue_indiscriminate(&qp->cq, cq_slot, &qp->sq, cq_state,
                              claimed_pos, wait_from);

    put_cid(&qp->sq, cpl_cid);
}

#endif // __NVM_QUEUE_POLICY_H__
