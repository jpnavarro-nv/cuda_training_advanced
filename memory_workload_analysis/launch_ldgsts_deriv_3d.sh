# Recompile code
cd build && make -j8 && cd ..

# Launch NCU profiler
ncu --set full --clock-control none -k regex:deriv --force-overwrite -c 5 -o profiles/ldgsts_deriv_3d \
./build/bin/ldgsts_deriv_3d n0=32 n1=16 n2=64