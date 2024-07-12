cmake_minimum_required(VERSION 3.0)
project(KokkosTest)

# Find Kokkos package
find_package(Kokkos REQUIRED)

# Add your executable
add_executable(test test.cpp)

# Link Kokkos to your executable
target_link_libraries(test Kokkos::kokkos)
