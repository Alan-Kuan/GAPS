add_executable(test0)
target_sources(test0 PRIVATE test0.cu)
target_include_directories(test0 PRIVATE .)
target_link_libraries(test0
    PRIVATE
        CUDA::cuda_driver
        zenohcxx::zenohc::lib
        node
        allocator
)

add_executable(test0-std)
target_sources(test0-std PRIVATE test0-std.cu)
target_include_directories(test0-std PRIVATE .)
target_link_libraries(test0-std
    PRIVATE
        CUDA::cuda_driver
        zenohcxx::zenohc::lib
        node
        allocator
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/script/* DESTINATION ${CMAKE_CURRENT_BINARY_DIR})