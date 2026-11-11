/* W-broadcast microbenchmark: is a dual-rail (2x half-size, disjoint
 * 2-rank comms) broadcast faster than one 4-rank row broadcast?
 * Launch with 16 ranks on 2 nodes (mirrors P4Q4); only row 0 measures. */
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
static int g_rank;
#define CK(x) do { auto e_ = (x); if (e_) { printf("err %d at %d\n", (int)e_, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1);} } while (0)
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int size; MPI_Comm_rank(MPI_COMM_WORLD, &g_rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm shm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm);
    int lrank; MPI_Comm_rank(shm, &lrank);
    int nd; cudaGetDeviceCount(&nd); cudaSetDevice(lrank % nd);
    const int P = 4, Q = size / P;
    const int pr = g_rank % P, qc = g_rank / P;
    /* full row comm (Q members) */
    MPI_Comm mrow; MPI_Comm_split(MPI_COMM_WORLD, pr, qc, &mrow);
    ncclComm_t nrow; { int rr; MPI_Comm_rank(mrow, &rr); ncclUniqueId id;
        if (!rr) ncclGetUniqueId(&id); MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mrow);
        CK(ncclCommInitRank(&nrow, Q, id, rr)); }
    /* half-row comms (even/odd qc) */
    MPI_Comm mrowH; MPI_Comm_split(mrow, qc & 1, qc, &mrowH);
    ncclComm_t nrowH; { int hr; MPI_Comm_rank(mrowH, &hr); ncclUniqueId id;
        if (!hr) ncclGetUniqueId(&id); MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mrowH);
        CK(ncclCommInitRank(&nrowH, Q / 2, id, hr)); }
    const size_t MB = (size_t)atoll(argv[1]) << 20; /* total W bytes */
    double *buf; CK(cudaMalloc(&buf, MB));
    cudaStream_t s1, s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const size_t n = MB / 8, half = n / 2;
    float ms;
    for (int rep = 0; rep < 3; rep++) {
        /* A: one broadcast over the full row */
        MPI_Barrier(MPI_COMM_WORLD);
        cudaEventRecord(e0, s1);
        for (int it = 0; it < 5; it++)
            CK(ncclBroadcast(buf, buf, n, ncclDouble, 0, nrow, s1));
        cudaEventRecord(e1, s1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        if (g_rank == 0) printf("rep%d A full-row  1x%zuMB: %7.2f ms/bcast  %6.1f GB/s\n",
                                rep, MB >> 20, ms / 5, MB / (ms / 5) / 1e6);
        /* B: two half broadcasts on disjoint parity comms, parallel streams.
         * roots: even rail root=q0 (hr 0), odd rail root=q1 (hr 0). */
        MPI_Barrier(MPI_COMM_WORLD);
        cudaEventRecord(e0, s1);
        for (int it = 0; it < 5; it++) {
            /* each rank participates in ONE parity comm */
            CK(ncclBroadcast(buf + (qc & 1) * half, buf + (qc & 1) * half,
                             half, ncclDouble, 0, nrowH, s1));
        }
        cudaEventRecord(e1, s1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        MPI_Barrier(MPI_COMM_WORLD);
        if (g_rank == 0) printf("rep%d B dual-rail 2x%zuMB: %7.2f ms/bcast  %6.1f GB/s eff\n",
                                rep, MB >> 21, ms / 5, MB / (ms / 5) / 1e6);
    }
    MPI_Finalize(); return 0;
}
