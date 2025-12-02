# Recompile code
cd build && make -j8 && cd ..

# Number of warps to launch - feel free to modify this value
nwarp=1
l2=64

# Launch NCU profiler
# Make sure to set the kernel count to 13 (there are 13 kernels to profile)  
ncu --set full --clock-control none -k regex:stride --force-overwrite -c 13 -o profiles/ldg_stride_l2_${l2}_nwarp_${nwarp} \
./build/bin/ldg_stride nwarp=${nwarp} l2=${l2}
