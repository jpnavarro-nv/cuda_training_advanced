# Recompile code
cd build && make -j8 && cd ..

# Launch NCU to profile code
ncu --set full --clock-control none -k regex:copy --force-overwrite -c 4 -o profiles/ldg_batch_nwarp1 \
./build/bin/ldg_batch_nwarp1 
ncu --set full --clock-control none -k regex:copy --force-overwrite -c 4 -o profiles/ldg_batch_nwarp10 \
./build/bin/ldg_batch_nwarp10 


