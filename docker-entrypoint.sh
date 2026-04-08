#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    set -- -np 2 mpi_cuda_bench
fi

if [[ "$1" == -* ]]; then
    exec mpirun \
        --mca pml ucx \
        --mca btl ^vader,tcp,openib \
        "$@"
fi

exec "$@"
