# Recompile code
cd build && make -j8 && cd ..

# Launch NCU profiler
#ncu --set full --clock-control none -k regex:ldgsts --force-overwrite -c 3 -o profiles/ldgsts_align_src16_dst0 \
# ./build/bin/ldgsts_align

# ncu --set full --clock-control none -k regex:ldgsts --force-overwrite -c 3 -o profiles/ldgsts_align_src16_dst0 \

# Commands to make performance = f(alignmment) figures
rm performance/perf_gmem.txt
./build/bin/ldgsts_align_perf perf_results=performance/perf_gmem.txt nval_gmem=9 nval_shmem=1

rm performance/perf_shmem.txt
./build/bin/ldgsts_align_perf perf_results=performance/perf_shmem.txt nval_gmem=1 nval_shmem=9

rm performance/perf_matrix.txt
./build/bin/ldgsts_align_perf perf_results=performance/perf_matrix.txt nval_gmem=9 nval_shmem=9
