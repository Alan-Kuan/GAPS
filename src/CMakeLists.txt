add_executable(${EXE})
target_sources(${EXE} PRIVATE main.cpp)
target_link_libraries(${EXE} PRIVATE comm)

add_executable(shareable_test)
target_sources(shareable_test PRIVATE shareable_test.cu)
target_link_libraries(shareable_test PRIVATE allocator error CUDA::cuda_driver)

add_library(allocator STATIC)
target_sources(allocator PRIVATE allocator/base.cpp allocator/shareable.cu)

add_library(comm STATIC)
target_sources(comm PRIVATE pub.cpp sub.cpp)
target_link_libraries(comm
    PUBLIC zenohcxx::zenohc::lib
    PRIVATE work)

add_library(error STATIC)
target_sources(error PRIVATE error.cpp)

add_library(work STATIC)
target_sources(work PRIVATE work.cu)