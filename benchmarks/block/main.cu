#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <ctrl.h>
#include <buffer.h>
#include "settings.h"
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <byteswap.h>
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;



//uint32_t n_ctrls = 1;
const char* const sam_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};
const char* const intel_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};
//const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9", "/dev/libnvm10", "/dev/libnvm11", "/dev/libnvm12", "/dev/libnvm13", "/dev/libnvm14", "/dev/libnvm15", "/dev/libnvm16", "/dev/libnvm17", "/dev/libnvm18", "/dev/libnvm19", "/dev/libnvm20", "/dev/libnvm21", "/dev/libnvm22", "/dev/libnvm23", "/dev/libnvm24","/dev/libnvm25", "/dev/libnvm26", "/dev/libnvm27", "/dev/libnvm28", "/dev/libnvm29", "/dev/libnvm30", "/dev/libnvm31"};

static const char* const profile_interval_names[] = {
    "read_total",
    "cid_acquire",
    "cmd_build",
    "sq_enqueue_total",
    "sq_ticket_wait",
    "sq_cmd_copy",
    "sq_tail_advance",
    "sq_mmio",
    "cq_poll_total",
    "cq_poll_scan",
    "cq_tail_atomic",
    "cq_dequeue_total",
    "cq_pos_lock",
    "cq_head_advance",
    "cq_mmio",
    "cq_visibility_wait",
    "cid_release"
};

static const size_t profile_interval_count = sizeof(profile_interval_names) / sizeof(profile_interval_names[0]);
static_assert(sizeof(nvm_read_profile_t) == profile_interval_count * sizeof(uint64_t), "profile row layout must match CSV columns");

static uint64_t profile_value(const nvm_read_profile_t& row, size_t idx)
{
    const uint64_t* values = reinterpret_cast<const uint64_t*>(&row);
    return values[idx];
}

static void write_profile_csv(const char* path, const std::vector<nvm_read_profile_t>& rows, uint64_t n_threads, uint64_t reqs_per_thread)
{
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out)
        throw error(std::string("Failed to open profile CSV: ") + path);

    out << "thread,iter";
    for (size_t i = 0; i < profile_interval_count; i++)
        out << "," << profile_interval_names[i];
    out << "\n";

    std::vector<double> sums(profile_interval_count, 0.0);
    std::vector<double> sum_squares(profile_interval_count, 0.0);
    const double row_count = static_cast<double>(rows.size());

    for (uint64_t tid = 0; tid < n_threads; tid++) {
        for (uint64_t iter = 0; iter < reqs_per_thread; iter++) {
            const nvm_read_profile_t& row = rows[tid * reqs_per_thread + iter];
            out << tid << "," << iter;
            for (size_t col = 0; col < profile_interval_count; col++) {
                uint64_t value = profile_value(row, col);
                sums[col] += static_cast<double>(value);
                sum_squares[col] += static_cast<double>(value) * static_cast<double>(value);
                out << "," << value;
            }
            out << "\n";
        }
    }

    out << std::fixed << std::setprecision(3);
    out << "AVERAGE,";
    for (size_t col = 0; col < profile_interval_count; col++) {
        double avg = row_count == 0.0 ? 0.0 : sums[col] / row_count;
        out << "," << avg;
    }
    out << "\n";

    out << "STDDEV,";
    for (size_t col = 0; col < profile_interval_count; col++) {
        double avg = row_count == 0.0 ? 0.0 : sums[col] / row_count;
        double variance = row_count == 0.0 ? 0.0 : (sum_squares[col] / row_count) - (avg * avg);
        if (variance < 0.0)
            variance = 0.0;
        out << "," << std::sqrt(variance);
    }
    out << "\n";
}


#define SIZE (8*4096)

__global__
void print_cache_kernel(page_cache_d_t* pc) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        hexdump(pc->base_addr, SIZE);
    }
}

__global__
void new_kernel(ulonglong4* dst, ulonglong4* src, size_t num) {
    warp_memcpy<ulonglong4>(dst, src, num);

}
/*
__device__ void read_data(page_cache_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    //printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) threadIdx.x, (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    sq_dequeue(&qp->sq, sq_pos);
    cq_dequeue(&qp->cq, cq_pos);


    put_cid(&qp->sq, cid);


}

*/
template <bool PROFILE>
__global__ __launch_bounds__(64,32)
void sequential_access_kernel(Controller** ctrls, page_cache_d_t* pc,  uint32_t req_size, uint32_t n_reqs, unsigned long long* req_count, uint32_t num_ctrls, uint64_t reqs_per_thread, uint32_t access_type, uint8_t* access_type_assignment, nvm_read_profile_t* profiles) {
    //printf("in threads\n");
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t laneid = lane_id();
    uint32_t bid = blockIdx.x;
    uint32_t smid = get_smid();

    uint32_t ctrl;
    uint32_t queue;
    if (laneid == 0) {
        ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
        //queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) %  (ctrls[ctrl]->n_qps);
        queue = smid % (ctrls[ctrl]->n_qps);
    }
    ctrl =  __shfl_sync(0xFFFFFFFF, ctrl, 0);
    queue =  __shfl_sync(0xFFFFFFFF, queue, 0);



    if (tid < n_reqs) {
        uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //start_block = tid;
        uint64_t n_blocks = req_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
        //printf("tid: %llu\tstart_block: %llu\tn_blocks: %llu\n", (unsigned long long) tid, (unsigned long long) start_block, (unsigned long long) n_blocks);
        uint8_t opcode;
        for (size_t i = 0; i < reqs_per_thread; i++) {
            if (access_type == MIXED) {
                opcode = access_type_assignment[tid];
                access_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid, opcode);
            }
            else if (access_type == READ) {
                if (PROFILE) {
                    nvm_read_profile_t* prof = &profiles[tid * reqs_per_thread + i];
                    read_data_profiled(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid, prof);
                }
                else {
                    read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
                }

            }
            else {
                write_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
            }
        }
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //__syncthreads();
        //read_data(pc, (ctrls[ctrl].d_qps)+(queue),start_block*2, n_blocks, tid);
        //printf("tid: %llu finished\n", (unsigned long long) tid);

    }

}

template <bool PROFILE>
__global__ //__launch_bounds__(64,32)
void random_access_kernel(Controller** ctrls, page_cache_d_t* pc,  uint32_t req_size, uint32_t n_reqs, unsigned long long* req_count, uint32_t num_ctrls, uint64_t* assignment, uint64_t reqs_per_thread, uint32_t access_type, uint8_t* access_type_assignment, nvm_read_profile_t* profiles) {
    //printf("in threads\n");
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t laneid = lane_id();
    uint32_t bid = blockIdx.x;
    uint32_t smid = get_smid();

    uint32_t ctrl;
    uint32_t queue;
    if (laneid == 0) {
	//ctrl = smid % (pc->n_ctrls);
        ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
        queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) %  (ctrls[ctrl]->n_qps);
        //queue = smid % (ctrls[ctrl]->n_qps);
    }
    ctrl =  __shfl_sync(0xFFFFFFFF, ctrl, 0);
    queue =  __shfl_sync(0xFFFFFFFF, queue, 0);


    if (tid < n_reqs) {
        uint64_t start_block = (assignment[tid]*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //start_block = tid;
        uint64_t n_blocks = req_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
        //printf("tid: %llu\tstart_block: %llu\tn_blocks: %llu\n", (unsigned long long) tid, (unsigned long long) start_block, (unsigned long long) n_blocks);

        uint8_t opcode;
        for (size_t i = 0; i < reqs_per_thread; i++) {
            if (access_type == MIXED) {
                opcode = access_type_assignment[tid];
                access_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid, opcode);
            }
            else if (access_type == READ) {
                if (PROFILE) {
                    nvm_read_profile_t* prof = &profiles[tid * reqs_per_thread + i];
                    read_data_profiled(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid, prof);
                }
                else {
                    read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
                }

            }
            else {
                write_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
            }
        }
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //__syncthreads();
        //read_data(pc, (ctrls[ctrl].d_qps)+(queue),start_block*2, n_blocks, tid);
        //printf("tid: %llu finished\n", (unsigned long long) tid);

    }

}


int main(int argc, char** argv) {

    Settings settings;
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }


    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    try {

        const char* input_f;

        input_f = settings.input;

        void* map_in;
        int fd_in;
        struct stat sb_in;

        if (input_f != nullptr) {
            if((fd_in = open(input_f, O_RDWR)) == -1){
                fprintf(stderr, "Input file cannot be opened\n");
                return 1;
            }

            fstat(fd_in, &sb_in);

            map_in = mmap(NULL, sb_in.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_in, 0);

            if((map_in == (void*)-1)){
                fprintf(stderr,"Input file map failed %d\n",map_in);
                return 1;
            }

            close(fd_in);
        }
        //Controller ctrl(settings.controllerPath, settings.nvmNamespace, settings.cudaDevice);
        
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(settings.ssdtype == 0 ? sam_ctrls_paths[i] : intel_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);

        //auto dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(64*1024*10, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);

        //std::cout << dma.get()->vaddr << std::endl;
        //QueuePair h_qp(ctrl, settings, 1);
        //std::cout << "in main: " << std::hex << h_qp.sq.cid << "raw: " << h_qp.sq.cid<< std::endl;
        //std::memset(&h_qp, 0, sizeof(QueuePair));
        //prepareQueuePair(h_qp, ctrl, settings, 1);
        //const uint32_t ps, const uint64_t np, const uint64_t c_ps, const Settings& settings, const Controller& ctrl)
        //
        /*
        Controller** d_ctrls;
        cuda_err_chk(cudaMalloc(&d_ctrls, n_ctrls*sizeof(Controller*)));
        for (size_t i = 0; i < n_ctrls; i++)
            cuda_err_chk(cudaMemcpy(d_ctrls+i, &(ctrls[i]->d_ctrl), sizeof(Controller*), cudaMemcpyHostToDevice));
        */
        uint64_t b_size = settings.blkSize;//64;
        uint64_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint64_t n_threads = b_size * g_size;


        uint64_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t total_cache_size = (page_size * n_pages);
        //uint64_t n_pages = total_cache_size/page_size;
        //
        if (n_pages < n_threads) {
            std::cerr << "Please provide enough pages. Number of pages must be greater than or equal to the number of threads!\n";
            exit(1);
        }


        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating cache\n";

        //QueuePair* d_qp;
        page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc.d_pc_ptr);
        #define TYPE uint64_t
        uint64_t n_blocks = settings.numBlks;
        //uint64_t t_size = n_blocks * sizeof(TYPE);

        // range_t<uint64_t> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)(t_size/page_size), (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        // range_t<uint64_t>* d_range = (range_t<uint64_t>*) h_range.d_range_ptr;

        // std::vector<range_t<uint64_t>*> vr(1);
        // vr[0] = & h_range;
        // //(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings)
        // array_t<uint64_t> a(n_elems, 0, vr, settings.cudaDevice);


        //std::cout << "finished creating range\n";




        unsigned long long* d_req_count;
        cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
        cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));

        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;
        uint64_t* assignment;
        uint64_t* d_assignment;
        if (settings.random) {
            assignment = (uint64_t*) malloc(n_threads*sizeof(uint64_t));
            for (size_t i = 0; i< n_threads; i++)
                assignment[i] = rand() % (n_blocks);


            cuda_err_chk(cudaMalloc(&d_assignment, n_threads*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_assignment, assignment,  n_threads*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
        Event before;

        uint8_t* access_assignment;
        uint8_t* d_access_assignment = NULL;
        if (settings.accessType == 2) {
            access_assignment = (uint8_t*) malloc(n_threads*sizeof(uint8_t));
            for (size_t i = 0; i < n_threads; i++)
                access_assignment[i] = (((rand() % 100) + 1) <= settings.ratio) ? NVM_IO_READ : NVM_IO_WRITE;

            cuda_err_chk(cudaMalloc(&d_access_assignment, n_threads*sizeof(uint8_t)));
            cuda_err_chk(cudaMemcpy(d_access_assignment, access_assignment, n_threads*sizeof(uint8_t), cudaMemcpyHostToDevice));
        }
        nvm_read_profile_t* d_profiles = NULL;
        const bool profile_enabled = settings.profileCsv != nullptr && settings.accessType == READ;
        const uint64_t profile_rows = n_threads * settings.numReqs;
        if (settings.profileCsv != nullptr && settings.accessType != READ) {
            std::cerr << "read_data profiling is only recorded for access_type=0; skipping profile CSV\n";
        }
        if (profile_enabled) {
            cuda_err_chk(cudaMalloc(&d_profiles, profile_rows * sizeof(nvm_read_profile_t)));
            cuda_err_chk(cudaMemset(d_profiles, 0, profile_rows * sizeof(nvm_read_profile_t)));
        }
        std::cout << "atlaunch kernel\n";
        if (settings.random) {
            if (profile_enabled)
                random_access_kernel<true><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType, d_access_assignment, d_profiles);
            else
                random_access_kernel<false><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType, d_access_assignment, d_profiles);
        }
        else {
            if (profile_enabled)
                sequential_access_kernel<true><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, settings.numReqs, settings.accessType, d_access_assignment, d_profiles);
            else
                sequential_access_kernel<false><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, settings.numReqs, settings.accessType, d_access_assignment, d_profiles);
        }
        Event after;

        //print_cache_kernel<<<1,1>>>(d_pc);
        //new_kernel<<<1,1>>>();
        //uint8_t* ret_array = (uint8_t*) malloc(n_pages*page_size);

        //cuda_err_chk(cudaMemcpy(ret_array, h_pc.base_addr,page_size*n_pages, cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaDeviceSynchronize());
        if (profile_enabled) {
            std::vector<nvm_read_profile_t> profiles(profile_rows);
            cuda_err_chk(cudaMemcpy(profiles.data(), d_profiles, profile_rows * sizeof(nvm_read_profile_t), cudaMemcpyDeviceToHost));
            write_profile_csv(settings.profileCsv, profiles, n_threads, settings.numReqs);
            cuda_err_chk(cudaFree(d_profiles));
        }
        if (input_f != nullptr) {
            cuda_err_chk(cudaMemcpy(map_in, h_pc.pdt.base_addr,  std::min((uint64_t)sb_in.st_size, total_cache_size), cudaMemcpyDeviceToHost));
            munmap(map_in, sb_in.st_size);
        }

        double elapsed = after - before;
        uint64_t ios = g_size*b_size*settings.numReqs;
        uint64_t data = ios*page_size;
        double iops = ((double)ios)/(elapsed/1000000);
        double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
        std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;
        h_pc.print_reset_stats();
        //std::cout << std::dec << ctrls[0]->ns.lba_data_size << std::endl;

        //std::ofstream ofile("../data", std::ios::binary | std::ios::trunc);
        //ofile.write((char*)ret_array, data);
        //ofile.close();

        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];
        //hexdump(ret_array, n_pages*page_size);
/*
        cudaFree(d_qp);
        cudaFree(d_pc);
        cudaFree(d_req_count);
        free(ret_array);
*/

        //std::cout << "END\n";

        //std::cout << RAND_MAX << std::endl;

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
