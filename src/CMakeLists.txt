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
        ticket_lock
        error)

add_library(allocator STATIC)
target_sources(allocator
    PRIVATE
        allocator/allocator.cpp
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
target_link_libraries(alloc_algo
    PRIVATE
        ticket_lock)

add_library(ticket_lock STATIC)
target_sources(ticket_lock
    PRIVATE
        ticket_lock.cpp)

add_executable(mem_manager)
target_sources(mem_manager
    PRIVATE
        mem_manager.cpp)
target_link_libraries(mem_manager
    PRIVATE
        CUDA::cuda_driver
        error)

add_library(error STATIC)
target_sources(error PRIVATE error.cpp)

# if(DEFINED BUILD_EXAMPLES)
    add_executable(cross_process)
    target_sources(cross_process
        PRIVATE
            examples/cross_process.cu)
    target_link_libraries(cross_process
        PRIVATE
            CUDA::cuda_driver
            zenohcxx::zenohc::lib
            vector_arithmetic
            node
            error)

    add_library(vector_arithmetic STATIC)
    target_sources(vector_arithmetic
        PRIVATE
            examples/vector_arithmetic.cu)
# endif()