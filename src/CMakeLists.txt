# Shoz
shoz_add_target(node
    SRCS node/node.cpp node/publisher.cpp node/subscriber.cpp
    LIBS CUDA::cuda_driver zenohcxx::zenohpico allocator error)
shoz_add_target(allocator
    SRCS allocator.cpp
    LIBS CUDA::cuda_driver alloc_algo error
    PIC)
shoz_add_target(alloc_algo
    SRCS alloc_algo/tlsf.cpp
    PIC)
shoz_add_target(error
    SRCS error.cpp
    PIC)

# Memory Manager
shoz_add_target(mem_manager
    IS_EXE
    SRCS mem_manager.cpp
    LIBS CUDA::cuda_driver error)

# Pyshoz
if(BUILD_TORCH_SUPPORT)
    nanobind_add_module(pyshoz
        pyshoz.cpp
        node/node.cpp
        node/publisher.cpp
        node/subscriber.cpp)
    target_compile_definitions(pyshoz
        PRIVATE
        BUILD_PYSHOZ)
    target_link_libraries(pyshoz
        PRIVATE
        CUDA::cuda_driver
        zenohcxx::zenohpico
        allocator
        alloc_algo
        error)
endif()