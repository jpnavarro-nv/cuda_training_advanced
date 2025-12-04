# Recompile code
cd build && make -j8 && cd ..

# Launch NCU profiler
ncu --set full --clock-control none -k regex:copy_shmem --force-overwrite -c 6 -o profiles/ldgsts_copy_shmem_stride \
./build/bin/ldgsts_copy kernel_type=ldg_sts
