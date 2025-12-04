# Recompile code
cd build && make -j8 && cd ..

# Launch NCU profiler
ncu --set full --clock-control none -k regex:deriv1d --force-overwrite -c 3 -o profiles/ldgsts_deriv_1d \
./build/bin/ldgsts_deriv_1d

