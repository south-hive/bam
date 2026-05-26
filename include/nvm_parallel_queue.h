#ifndef __NVM_PARALLEL_QUEUE_H_
#define __NVM_PARALLEL_QUEUE_H_

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "interval_profile.h"
#include <simt/atomic>
#define LOCKED   1
#define UNLOCKED 0

__forceinline__ __device__ uint64_t get_id(uint64_t x, uint64_t y) {
    return (x >> y) * 2;  // (x/2^y) *2
}

inline __device__
uint16_t get_cid(nvm_queue_t* sq) {
    bool not_found = true;
    uint16_t id;

    do {
        id = sq->cid_ticket.fetch_add(1, simt::memory_order_relaxed) & (65535);
        uint64_t old = sq->cid[id].val.fetch_or(LOCKED, simt::memory_order_acquire);
        not_found = old == LOCKED;
    } while (not_found);

    return id;

}

inline __device__
void put_cid(nvm_queue_t* sq, uint16_t id) {
    sq->cid[id].val.store(UNLOCKED, simt::memory_order_release);
}

inline __device__
uint32_t move_tail(nvm_queue_t* q, uint32_t cur_tail) {
    uint32_t count = 0;

    bool pass = true;
    while (pass ) {
        pass = (((cur_tail+count+1) & q->qs_minus_1) != (q->head.load(simt::memory_order_relaxed) & q->qs_minus_1 ));
        if (pass) {
            pass = ((q->tail_mark[(cur_tail+count)&q->qs_minus_1].val.exchange(UNLOCKED, simt::memory_order_relaxed)) == LOCKED);
            if (pass)
                count++;
        }

    }

    q->head_lock.fetch_add(1, simt::memory_order_acq_rel);
    return (count);
}

inline __device__
uint32_t move_head_cq(nvm_queue_t* q, uint32_t cur_head, nvm_queue_t* sq) {
    uint32_t count = 0;
    (void) sq;

    bool pass = true;
    while (pass) {
        uint32_t loc = (cur_head+count++)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.exchange(UNLOCKED, simt::memory_order_relaxed)) == LOCKED;
    }
    count -= 1;
    if (count) {
        uint32_t loc_ = (cur_head + (count -1)) & q->qs_minus_1;
        uint32_t cpl_entry = ((nvm_cpl_t*)q->vaddr)[loc_].dword[2];
        uint16_t new_sq_head =  (cpl_entry & 0x0000ffff);
        uint32_t sq_move_count = 0;
        uint32_t cur_sq_head = sq->head.load(simt::memory_order_relaxed);
        uint32_t loc = cur_sq_head & sq->qs_minus_1;

        if (loc != new_sq_head) {
            for (; loc != new_sq_head; sq_move_count++, loc= ((loc+1)  & sq->qs_minus_1)) {
                sq->tickets[loc].val.fetch_add(1, simt::memory_order_relaxed);
            }

            sq->head.fetch_add(sq_move_count, simt::memory_order_acq_rel);
        }
    }
    return (count);

}

inline __device__
void clean_cids(nvm_queue_t* cq, nvm_queue_t* sq, uint32_t count) {
    for (size_t i  = 0; i < count; i++) {
        put_cid(sq, cq->clean_cid[i]);
    }
}

inline __device__
uint32_t move_head_sq(nvm_queue_t* q, uint32_t cur_head) {
    uint32_t count = 0;

    bool pass = true;
    while (pass) {
        uint64_t loc = (cur_head + count)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.exchange(UNLOCKED, simt::memory_order_relaxed)) == LOCKED;
        if (pass) {
            q->tickets[loc].val.fetch_add(1, simt::memory_order_relaxed);
            count++;
        }
    }
    return (count);
}

typedef ulonglong4_32a copy_type;

inline __device__
uint16_t sq_enqueue(nvm_queue_t* sq, nvm_cmd_t* cmd, simt::atomic<uint64_t, simt::thread_scope_device>* pc_tail =NULL, uint64_t * cur_pc_tail=NULL, interval_record_t* rec = nullptr) {

    uint32_t ticket;
    ticket = sq->in_ticket.fetch_add(1, simt::memory_order_relaxed);

    uint32_t pos = ticket & (sq->qs_minus_1);
    uint64_t id = get_id(ticket, sq->qs_log2);

    unsigned int ns = 8;
    while ((sq->tickets[pos].val.load(simt::memory_order_relaxed) != id) ) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    ns = 8;
    while ((sq->tickets[pos].val.load(simt::memory_order_acquire) != id) ) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    PROF_STAMP(rec, PROF_PH_SQ_TICKET_WAIT);

    copy_type* queue_loc = ((copy_type*)(((nvm_cmd_t*)(sq->vaddr)) + pos));
    copy_type* cmd_ = ((copy_type*)(cmd->dword));

#pragma unroll
    for (uint32_t i = 0; i < 64/sizeof(copy_type); i++) {
        queue_loc[i] = cmd_[i];
    }

    PROF_STAMP(rec, PROF_PH_SQ_CMD_COPY);

    if (pc_tail) {
        *cur_pc_tail = pc_tail->load(simt::memory_order_relaxed);
    }
    sq->tail_mark[pos].val.store(LOCKED, simt::memory_order_release);
    bool cont = true;
    ns = 8;
    cont = sq->tail_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
    while(cont) {
        bool new_cont = sq->tail_lock.load(simt::memory_order_relaxed) == LOCKED;
        if (!new_cont) {
            new_cont = sq->tail_lock.fetch_or(LOCKED, simt::memory_order_acquire) == LOCKED;
            if(!new_cont) {
                uint32_t cur_tail = sq->tail.load(simt::memory_order_relaxed);

                uint32_t tail_move_count = move_tail(sq, cur_tail);

                if (tail_move_count) {
                    uint32_t new_tail = cur_tail + tail_move_count;
                    uint32_t new_db = (new_tail) & (sq->qs_minus_1);
                    if (pc_tail) {
                        *cur_pc_tail = pc_tail->load(simt::memory_order_acquire);
                    }
		    asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(sq->db),"r"(new_db) : "memory");

                    sq->tail.store(new_tail, simt::memory_order_release);
                }
                sq->tail_lock.store(UNLOCKED, simt::memory_order_release);
            }
        }
        cont = sq->tail_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
        if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
        }

    }

    sq->tickets[pos].val.fetch_add(1, simt::memory_order_acq_rel);
    PROF_STAMP(rec, PROF_PH_SQ_TAIL_ADVANCE);
    return pos;

}

inline __device__
void sq_dequeue(nvm_queue_t* sq, uint16_t pos) {

    sq->head_mark[pos].val.store(LOCKED, simt::memory_order_relaxed);
    bool cont = true;
    unsigned int ns = 8;
    cont = sq->head_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
    while (cont) {
            bool new_cont = sq->head_lock.exchange(LOCKED, simt::memory_order_acquire) == LOCKED;
            if (!new_cont){
                uint32_t cur_head = sq->head.load(simt::memory_order_relaxed);;

                uint32_t head_move_count = move_head_sq(sq, cur_head);
                if (head_move_count) {
                    sq->head.store(cur_head + head_move_count, simt::memory_order_relaxed);
                }

                sq->head_lock.store(UNLOCKED, simt::memory_order_release);
            }
            cont = sq->head_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
            if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                __nanosleep(ns);
                if (ns < 256) {
                    ns *= 2;
                }
#endif

            }
    }

}

inline __device__
uint32_t cq_poll(nvm_queue_t* cq, uint16_t search_cid, uint32_t* loc_ = NULL, uint32_t* cq_head = NULL, interval_record_t* rec = nullptr) {
    uint64_t j = 0;
    unsigned int ns = 8;
    while (true) {
        uint32_t head = cq->head.load(simt::memory_order_relaxed);

        for (size_t i = 0; i < cq->qs_minus_1; i++) {
            uint32_t cur_head = head + i;
            bool search_phase = ((~(cur_head >> cq->qs_log2)) & 0x01);
            uint32_t loc = cur_head & (cq->qs_minus_1);
            uint32_t cpl_entry = ((nvm_cpl_t*)cq->vaddr)[loc].dword[3];
            uint32_t cid = (cpl_entry & 0x0000ffff);
            bool phase = (cpl_entry & 0x00010000) >> 16;
            if ((cid == search_cid) && (phase == search_phase)){
                *cq_head = head;
                *loc_ = cur_head;
                PROF_STAMP(rec, PROF_PH_CQ_POLL_SCAN);
                return loc;
            }
            if (phase != search_phase)
                break;
        }
        j++;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) {
             ns *= 2;
         }
#endif
    }
}

inline __device__
void cq_dequeue(nvm_queue_t* cq, uint16_t pos, nvm_queue_t* sq, uint32_t loc_ = 0, uint32_t cur_head_ = 0, interval_record_t* rec = nullptr) {
    cq->tail.fetch_add(1, simt::memory_order_acq_rel);

    unsigned int ns = 8;
    while ((cq->pos_locks[pos].val.load(simt::memory_order_relaxed) != 0) ) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    ns = 8;
    while ((cq->pos_locks[pos].val.fetch_or(1, simt::memory_order_acquire) != 0) ) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    cq->head_mark[pos].val.store(LOCKED, simt::memory_order_release);

    bool cont = true;
    ns = 8;
    cont = cq->head_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
    while (cont) {
            bool new_cont = cq->head_lock.fetch_or(LOCKED, simt::memory_order_acquire) == LOCKED;
            if (!new_cont) {
                uint32_t cur_head = cq->head.load(simt::memory_order_relaxed);;

                uint32_t head_move_count = move_head_cq(cq, cur_head, sq);

                if (head_move_count) {
                    uint32_t new_head = cur_head + head_move_count;

                    uint32_t new_db = (new_head) & (cq->qs_minus_1);

                    asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(cq->db),"r"(new_db) : "memory");

                    cq->head.store(new_head, simt::memory_order_release);
                }
                cq->head_lock.store(UNLOCKED, simt::memory_order_release);
            }
            cont = cq->head_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
            if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                __nanosleep(ns);
                if (ns < 256) {
                    ns *= 2;
                }
#endif
            }
    }

	uint64_t j = 0;
    uint32_t new_head = cq->head.load(simt::memory_order_relaxed);
    ns = 8;
    do {
        if (new_head > cur_head_) {
            if ((loc_ >= cur_head_) && (loc_ < new_head))
                break;

        }
        else if (new_head < cur_head_) {
            if ((loc_ >= cur_head_))
                break;
            if (loc_ < new_head)
                break;
        }

        j++;
        new_head = cq->head.load(simt::memory_order_relaxed);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    } while(true);

    cq->pos_locks[pos].val.store(0, simt::memory_order_release);
    PROF_STAMP(rec, PROF_PH_CQ_HEAD_ADVANCE);
}

#endif // __NVM_PARALLEL_QUEUE_H_
