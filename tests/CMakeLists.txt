add_executable(test0)
target_sources(test0 PRIVATE test0.cu)
target_include_directories(test0 PRIVATE .)
target_link_libraries(test0
    PRIVATE
        CUDA::cuda_driver
        zenohcxx::zenohpico
        node
        allocator
)

add_executable(test0-p2p)
target_sources(test0-p2p PRIVATE test0-p2p.cu)
target_include_directories(test0-p2p PRIVATE .)
target_link_libraries(test0-p2p
    PRIVATE
        CUDA::cuda_driver
        zenohcxx::zenohpico
        node
        allocator
)