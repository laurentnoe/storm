cmake_minimum_required (VERSION 2.6)
project (SToRM)


#set(CMAKE_C_FLAGS "-Wall -Wextra -pedantic -lm -msse2 -O3")
set(CMAKE_C_FLAGS "-Wall -Wextra -lm -msse2 -O3")

# large files

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE)

# OpenMP

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# storm-color

add_executable(storm-color main.c)
add_library(lib-storm-color alignment.c alignment_simd.c codes.c hit_map.c index.c load_data.c genome_map.c reads_data.c read_quality.c reference_data.c run.c seed.c stats.c util.c)
target_link_libraries(storm-color lib-storm-color)

# storm-nucleotide

add_executable(storm-nucleotide main.c)
add_library(lib-storm-nucleotide alignment.c alignment_simd.c codes.c hit_map.c index.c load_data.c genome_map.c reads_data.c read_quality.c reference_data.c run.c seed.c stats.c util.c)
target_link_libraries(storm-nucleotide lib-storm-nucleotide)
target_compile_definitions(storm-nucleotide PUBLIC -DNUCLEOTIDES)
target_compile_definitions(lib-storm-nucleotide PUBLIC -DNUCLEOTIDES)

# installation

install(TARGETS storm-color storm-nucleotide RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
