cmake_minimum_required(VERSION 3.18) 
project(SEGMERGE LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES ${ARCH})  # Set CUDA architectures

add_executable(run main.cu)

target_compile_features(run PUBLIC cxx_std_17)

# Set CUDA flags
set_target_properties(run PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "86"
)

# Define include directories
#target_include_directories(run PRIVATE ${CMAKE_SOURCE_DIR}/GSAP)

# Specify compiler options for CUDA and C++
target_compile_options(run PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -O3
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
  >
)

# Add a custom target for cleaning up that doesn't use the reserved name "clean"
add_custom_target(cleanall
  COMMAND ${CMAKE_COMMAND} -E remove run
)

# Enable testing
enable_testing()

# Add tests with different argument combinations
add_test(NAME Test_n_128 COMMAND run 128)
add_test(NAME Test_n_256 COMMAND run 256)
add_test(NAME Test_n_512 COMMAND run 512)
add_test(NAME Test_n_1024 COMMAND run 1024)
add_test(NAME Test_n_2048 COMMAND run 2048)
add_test(NAME Test_n_4096 COMMAND run 4096)
add_test(NAME Test_n_8192 COMMAND run 8192)
add_test(NAME Test_n_16384 COMMAND run 16384)
add_test(NAME Test_n_32768 COMMAND run 32768)
add_test(NAME Test_n_65536 COMMAND run 65536)
add_test(NAME Test_n_131072 COMMAND run 131072)


