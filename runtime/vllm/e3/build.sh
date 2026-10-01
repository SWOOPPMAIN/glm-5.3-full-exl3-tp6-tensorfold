#!/bin/sh
set -eu
cd "$(dirname "$0")"
nvcc -O3 --fmad=false -lineinfo --cubin -arch=sm_121a amos_e3/grouped_fragments.cu -o amos_e3/grouped_fragments.cubin
nvcc --version
sha256sum amos_e3/grouped_fragments.cubin
