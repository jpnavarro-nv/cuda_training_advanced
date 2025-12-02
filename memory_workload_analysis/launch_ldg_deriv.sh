# Recompile code
cd build && make -j8 && cd ..

##################### Copy kernels #####################
# ncu --set full --clock-control none -k regex:ldg_deriv_copy --force-overwrite -c 2 -o profiles/ldg_copy_n0_32_n1_1 \
# ./build/bin/ldg_deriv_1d n0=32 n1=1 kernel=copy

ncu --set full --clock-control none -k regex:ldg_deriv_copy --force-overwrite -c 3 -o profiles/ldg_copy_n0_32_n1_2 \
./build/bin/ldg_deriv_1d n0=32 n1=2 kernel=copy
ncu --set full --clock-control none -k regex:ldg_deriv_copy --force-overwrite -c 3 -o profiles/ldg_copy_n0_64_n1_1 \
./build/bin/ldg_deriv_1d n0=64 n1=1 kernel=copy

# ncu --set full --clock-control none -k regex:ldg_deriv_copy --force-overwrite -c 3 -o profiles/ldg_copy_n0_1024_n1_1 \
# ./build/bin/ldg_deriv_1d n0=1024 n1=1 kernel=copy
# ncu --set full --clock-control none -k regex:ldg_deriv_copy --force-overwrite -c 3 -o profiles/ldg_copy_n0_32_n1_2 \
# ./build/bin/ldg_deriv_1d n0=32 n1=2 kernel=copy
# ncu --set full --clock-control none -k regex:ldg_deriv_copy --force-overwrite -c 3 -o profiles/ldg_copy_n0_32_n1_5 \
# ./build/bin/ldg_deriv_1d n0=32 n1=5 kernel=copy

# Uncomment this to modify array dimensions
# n0=64
# n1=1
# ncu --set full --clock-control none -k regex:ldg_deriv_copy --force-overwrite -c 3 -o profiles/ldg_copy_n0_${n0}_n1_${n1} \
# ./build/bin/ldg_deriv_1d n0=${n0} n1=${n1} kernel=copy

##################### Derivative kernel in leading dimension #####################
# ncu --set full --clock-control none -k regex:ldg_deriv0 --force-overwrite -c 3 -o profiles/ldg_deriv0_n0_32_n1_1 \
# ./build/bin/ldg_deriv_1d n0=32 n1=1 kernel=deriv0

# ##################### Derivative kernel in slow dimension #####################
# # Derivative in the "slow" dimension
# ncu --set full --clock-control none -k regex:ldg_deriv1 --force-overwrite -c 3 -o profiles/ldg_deriv1_n0_${n0}_n1_${n1} \
# ./build/bin/ldg_deriv_1d n0=32 n1=1 kernel=deriv1

