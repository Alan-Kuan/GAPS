# Shoi
prj_add_target(node
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

# Pyshoi
if(BUILD_TORCH_SUPPORT)
    nanobind_add_module(pyshoi
        pyshoi.cpp
        node/node.cpp
        node/publisher.cpp
        node/subscriber.cpp)
    target_compile_definitions(pyshoi
        PRIVATE
        BUILD_PYSHOI)
    target_link_libraries(pyshoi
        PRIVATE
        CUDA::cuda_driver
        iceoryx_posh::iceoryx_posh
        allocator
        error)
endif()
