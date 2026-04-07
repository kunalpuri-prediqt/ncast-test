#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Simple kernel: first 4 threads print their IDs
// ---------------------------------------------------------------------------
__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 4) {
        printf("  [GPU] Hello from thread %d  (block %d, threadIdx %d)\n",
               tid, blockIdx.x, threadIdx.x);
    }
}

// ---------------------------------------------------------------------------
// Bandwidth benchmark — mirrors the Python/Warp template
// ---------------------------------------------------------------------------
static void benchmark(MPI_Comm comm, int rank, bool cuda_aware,
                       const int *sizes, int nsizes, int iterations) {
    if (rank == 0) {
        const char *mode = cuda_aware ? "CUDA-Aware (Zero-Copy)"
                                      : "Host-Staged (Manual)";
        printf("\nBenchmarking %s\n", mode);
        printf("%10s | %15s\n", "Size (MB)", "Bandwidth (GB/s)");
    }

    for (int s = 0; s < nsizes; ++s) {
        int n = sizes[s];                       // number of float elements
        size_t bytes = (size_t)n * sizeof(float);

        // GPU buffer
        float *d_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buf, bytes));
        CUDA_CHECK(cudaMemset(d_buf, 0, bytes));

        // Host staging buffer (used only when NOT cuda-aware)
        float *h_buf = nullptr;
        if (!cuda_aware) {
            CUDA_CHECK(cudaMallocHost(&h_buf, bytes));   // pinned for speed
            memset(h_buf, 0, bytes);
        }

        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        for (int i = 0; i < iterations; ++i) {
            if (cuda_aware) {
                // Direct GPU pointer through MPI
                if (rank == 0) {
                    MPI_Send(d_buf, n, MPI_FLOAT, 1, 10, comm);
                    MPI_Recv(d_buf, n, MPI_FLOAT, 1, 11, comm,
                             MPI_STATUS_IGNORE);
                } else if (rank == 1) {
                    MPI_Recv(d_buf, n, MPI_FLOAT, 0, 10, comm,
                             MPI_STATUS_IGNORE);
                    MPI_Send(d_buf, n, MPI_FLOAT, 0, 11, comm);
                }
            } else {
                // Manual D2H -> MPI -> H2D staging
                if (rank == 0) {
                    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes,
                                          cudaMemcpyDeviceToHost));
                    MPI_Send(h_buf, n, MPI_FLOAT, 1, 10, comm);
                    MPI_Recv(h_buf, n, MPI_FLOAT, 1, 11, comm,
                             MPI_STATUS_IGNORE);
                    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes,
                                          cudaMemcpyHostToDevice));
                } else if (rank == 1) {
                    MPI_Recv(h_buf, n, MPI_FLOAT, 0, 10, comm,
                             MPI_STATUS_IGNORE);
                    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes,
                                          cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes,
                                          cudaMemcpyDeviceToHost));
                    MPI_Send(h_buf, n, MPI_FLOAT, 0, 11, comm);
                }
            }
        }

        double t1 = MPI_Wtime();

        if (rank == 0) {
            double total_gb = (2.0 * n * sizeof(float) * iterations) / 1e9;
            double bw = total_gb / (t1 - t0);
            printf("%10.2f | %15.2f\n", bytes / 1e6, bw);
        }

        CUDA_CHECK(cudaFree(d_buf));
        if (h_buf) CUDA_CHECK(cudaFreeHost(h_buf));
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Each rank picks its own GPU (round-robin)
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(rank % ndev));

    // --- 1. Launch a tiny kernel and print from the first 4 threads ----------
    if (rank == 0) {
        printf("Rank %d: launching hello_kernel on GPU %d\n", rank, rank % ndev);
        hello_kernel<<<1, 32>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- 2. MPI bandwidth benchmark -----------------------------------------
    if (nranks < 2) {
        if (rank == 0)
            printf("\nNeed at least 2 MPI ranks for bandwidth test. "
                   "Run with: mpirun -np 2 ./mpi_cuda_bench\n");
        MPI_Finalize();
        return 0;
    }

    // Test sizes: 16 MB, 64 MB, 128 MB, 256 MB  (in number of float elements)
    const int MB = 1024 * 1024;
    int test_sizes[] = {
        (MB / 4) * 16,
        (MB / 4) * 64,
        (MB / 4) * 128,
        //(MB / 4) * 256,
    };
    int nsizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int iterations = 50;

    benchmark(MPI_COMM_WORLD, rank, /*cuda_aware=*/false,
              test_sizes, nsizes, iterations);

    benchmark(MPI_COMM_WORLD, rank, /*cuda_aware=*/true,
              test_sizes, nsizes, iterations);

    MPI_Finalize();
    return 0;
}
