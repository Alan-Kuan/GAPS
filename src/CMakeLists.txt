# Shoi
prj_add_target(shoi
    SRCS node/node.cpp node/publisher.cpp node/subscriber.cpp
    LIBS CUDA::cuda_driver iceoryx_posh::iceoryx_posh allocator error)
prj_add_target(allocator
    SRCS allocator/allocator.cpp allocator/tlsf.cpp
    LIBS CUDA::cuda_driver error
    PIC)
prj_add_target(error
    SRCS error.cpp
    PIC)

# Memory Manager
prj_add_target(mem_manager
    IS_EXE
    SRCS mem_manager.cpp
    LIBS CUDA::cuda_driver error)

# Pyshoz
# TODO: port shoi
if(BUILD_TORCH_SUPPORT)
    nanobind_add_module(pyshoz
        pyshoz.cpp
        node/node.cpp
        node/publisher.cpp
        node/subscriber.cpp
        zenoh_wrapper.cpp)
    target_compile_definitions(pyshoz
        PRIVATE
        BUILD_PYSHOZ)
    target_link_libraries(pyshoz
        PRIVATE
        CUDA::cuda_driver
        zenohcxx::zenohpico
        allocator
        error)
endif()
