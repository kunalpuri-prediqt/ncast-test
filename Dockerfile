# ============================================================================
# Dockerfile — ncast MPI+CUDA benchmark (reproducible build)
# ============================================================================
# Base: NVIDIA CUDA 13.2 devel on Ubuntu 22.04
# Builds: UCX 1.20.0 → OpenMPI 5.0.10 → mpi_cuda_bench
# ============================================================================
FROM nvidia/cuda:13.2.0-devel-ubuntu22.04 AS builder

ARG UCX_VERSION=1.20.0
ARG OMPI_VERSION=5.0.10
ARG PREFIX=/opt/ncast
ARG JOBS=8

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        automake \
        autoconf \
        libtool \
        flex \
        wget \
        ca-certificates \
        libibverbs-dev \
        librdmacm-dev \
        libnuma-dev \
        pkg-config \
        python3 \
        perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ---- UCX ------------------------------------------------------------------
RUN wget -q https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz \
    && tar xzf ucx-${UCX_VERSION}.tar.gz \
    && cd ucx-${UCX_VERSION} \
    && ./configure --prefix=${PREFIX} \
                   --with-cuda=/usr/local/cuda \
                   --enable-mt \
    && make -j${JOBS} \
    && make install \
    && cd .. && rm -rf ucx-${UCX_VERSION} ucx-${UCX_VERSION}.tar.gz

# ---- OpenMPI ---------------------------------------------------------------
RUN wget -q https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VERSION}.tar.gz \
    && tar xzf openmpi-${OMPI_VERSION}.tar.gz \
    && cd openmpi-${OMPI_VERSION} \
    && ./configure --prefix=${PREFIX} \
                   --with-ucx=${PREFIX} \
                   --with-ucx-libdir=${PREFIX}/lib \
                   --with-cuda=/usr/local/cuda \
                   --enable-mca-no-build=btl-uct \
    && make -j${JOBS} \
    && make install \
    && cd .. && rm -rf openmpi-${OMPI_VERSION} openmpi-${OMPI_VERSION}.tar.gz

# ---- Solver ----------------------------------------------------------------
COPY mpi_cuda_bench.cu /build/
COPY mca-params.conf   ${PREFIX}/etc/mca-params.conf

ENV PATH=${PREFIX}/bin:${PATH}
ENV LD_LIBRARY_PATH=${PREFIX}/lib:/usr/local/cuda/lib64
ENV C_INCLUDE_PATH=${PREFIX}/include:/usr/local/cuda/include
ENV CPLUS_INCLUDE_PATH=${PREFIX}/include:/usr/local/cuda/include

RUN nvcc -ccbin mpicxx -o ${PREFIX}/bin/mpi_cuda_bench /build/mpi_cuda_bench.cu \
        -I${PREFIX}/include -L${PREFIX}/lib -lmpi

# ============================================================================
# Runtime stage — smaller image
# ============================================================================
FROM nvidia/cuda:13.2.0-runtime-ubuntu22.04

ARG PREFIX=/opt/ncast

RUN apt-get update && apt-get install -y --no-install-recommends \
        libibverbs1 \
        librdmacm1 \
        libnuma1 \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder ${PREFIX} ${PREFIX}

ENV PATH=${PREFIX}/bin:${PATH}
ENV LD_LIBRARY_PATH=${PREFIX}/lib:/usr/local/cuda/lib64

RUN useradd -m -s /bin/bash ncast

WORKDIR /workspace
COPY mca-params.conf /workspace/mca-params.conf

# Directories for test I/O
RUN mkdir -p /workspace/test-input /workspace/test-output \
    && chown -R ncast:ncast /workspace

USER ncast

# Default: run the benchmark with 2 ranks on available GPUs
ENTRYPOINT ["mpirun", \
            "--mca", "pml", "ucx", \
            "--mca", "btl", "^vader,tcp,openib"]
CMD ["-np", "2", "mpi_cuda_bench"]
