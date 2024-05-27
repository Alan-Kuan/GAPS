add_library(node STATIC)
target_sources(node
    PRIVATE
        node/node.cpp
        node/publisher.cpp
        node/subscriber.cpp)
target_link_libraries(node
    PRIVATE
        zenohcxx::zenohc::lib
        allocator
        error)

add_library(allocator STATIC)
target_sources(allocator
    PRIVATE
        allocator/shareable.cu)
target_link_libraries(allocator
    PRIVATE
        CUDA::cuda_driver
        alloc_algo
        error)

add_library(alloc_algo STATIC)
target_sources(alloc_algo
    PRIVATE
        alloc_algo/tlsf.cpp)

add_library(error STATIC)
target_sources(error PRIVATE error.cpp)

# if(DEFINED BUILD_EXAMPLES)
    add_executable(cross_process)
    target_sources(cross_process PRIVATE examples/cross_process.cu)
    target_link_libraries(cross_process
        PRIVATE
            CUDA::cuda_driver
            zenohcxx::zenohc::lib
            vector_arithmetic
            node
            error)

    add_library(vector_arithmetic STATIC)
    target_sources(vector_arithmetic PRIVATE examples/vector_arithmetic.cu)
# endif()