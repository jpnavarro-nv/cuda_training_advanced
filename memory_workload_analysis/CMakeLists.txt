cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tutorial_shared_memory_barriers LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
if(CMAKE_VERBOSE_MAKEFILE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
endif()
option(FASTMATH "nvcc's `-use_fast_math` option" OFF)
if(FASTMATH)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
endif()
option(LINEINFO "nvcc's `-lineinfo` option" ON)
if(LINEINFO)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
option(KEEP "nvcc's `-keep` option" OFF)
if(KEEP)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-keep>)
endif()
# set(CMAKE_CUDA_ARCHITECTURES "80;90;100")
set(CMAKE_CUDA_ARCHITECTURES "80;90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message("--------------------------------> building: Tutorial Shared Memory Barriers")

# cmake -DCMAKE_BUILD_TYPE=Debug ..

#####################################################
#################### LDG (Lab1) #####################
#####################################################

#### Strided access
# LDG + strided access
add_executable(
  ldg_stride
  src/main_ldg_stride.cu
  src/parfileReader.cu
  )

target_include_directories(
  ldg_stride
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#### Batch copy
# LDG + 16 bytes, 1 warp
#add_executable(
#  ldg_batch_nwarp1
#  src/main_ldg_batch.cu
#  src/parfileReader.cu
#  )

#target_include_directories(
#  ldg_batch_nwarp1
#  PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#)

#target_compile_definitions(
#  ldg_batch_nwarp1
#  PRIVATE
#  NWARP=1
#)

# LDG + 16 bytes, 10 warps
#add_executable(
#  ldg_batch_nwarp10
#  src/main_ldg_batch.cu
#  src/parfileReader.cu
#  )

#target_include_directories(
#  ldg_batch_nwarp10
#  PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#)

#target_compile_definitions(
#  ldg_batch_nwarp10
#  PRIVATE
#  NWARP=10
#)

#### Finite-Difference operators
#add_executable(
#  ldg_deriv_1d
#  src/main_ldg_deriv_1d.cu
#  src/parfileReader.cu
#  )

#target_include_directories(
#  ldg_deriv_1d
#  PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#)

#####################################################
################# LDGSTS (Lab2) #####################
#####################################################
#### Copy kernels
#add_executable(
#  ldgsts_copy
#  src/main_ldgsts_copy.cu
#  src/parfileReader.cu
#  )

#target_include_directories(
#  ldgsts_copy
#  PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#)

#### Alignment effects
#add_executable(
#  ldgsts_align
#  src/main_ldgsts_align.cu
#  src/parfileReader.cu
# )

#target_include_directories(
#  ldgsts_align
#  PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#)

#### Alignment performance
#add_executable(
#  ldgsts_align_perf
#  src/main_ldgsts_align_perf.cu
#  src/parfileReader.cu
#  )

#target_include_directories(
#  ldgsts_align_perf
#  PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#)

#### Finite-Difference operators 1d input
#add_executable(
#  ldgsts_deriv_1d
#  src/main_ldgsts_deriv_1d.cu
#  src/parfileReader.cu
#  )

#target_include_directories(
#  ldgsts_deriv_1d
#  PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#)
#target_link_libraries(ldgsts_deriv_1d cuda curand)

#### Finite-Difference operators 3d input
#add_executable(
#  ldgsts_deriv_3d
#  src/main_ldgsts_deriv_3d.cu
#  src/parfileReader.cu
#  )

#target_include_directories(
#  ldgsts_deriv_3d
#  PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#)
#target_link_libraries(ldgsts_deriv_3d cuda curand)
