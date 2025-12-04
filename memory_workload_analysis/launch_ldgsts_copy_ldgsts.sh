# Recompile code
cd build && make -j8 && cd ..

# Launch NCU profiler
ncu --set full --clock-control none -k regex:copy_shmem --force-overwrite -c 4 -o profiles/ldgsts_copy_shmem_stride \
./build/bin/ldgsts_copy kernel_type=ldgsts

# ncu --set full --clock-control none -k regex:ldgsts --force-overwrite -c 3 -o profiles/ldgsts_align_src16_dst0 \
# ./build/bin/ldgsts_align
# ncu --set full --clock-control none -k regex:ldgsts --force-overwrite -c 3 -o profiles/ldgsts_align_src16_dst0 \
# rm perf_gmem.txt
# ./build/bin/ldgsts_align_perf perf_results=perf_gmem.txt nval_gmem=9 nval_shmem=1
# rm perf_shmem.txt
# ./build/bin/ldgsts_align_perf perf_results=perf_shmem.txt nval_gmem=1 nval_shmem=9
# rm perf_matrix.txt
# ./build/bin/ldgsts_align_perf perf_results=perf_matrix.txt nval_gmem=9 nval_shmem=9
# ./build/bin/ldgsts_compute n0=6
# ncu --set full --clock-control none -k regex:deriv1d --force-overwrite -c 3 -o profiles/ldgsts_deriv_1d_src_dst \
# ./build/bin/ldgsts_deriv_1d
# ncu --set full --clock-control none -k regex:deriv --force-overwrite -c 5 -o profiles/ldgsts_deriv_3d \
# ./build/bin/ldgsts_deriv_3d n0=32 n1=16 n2=2