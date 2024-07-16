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

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/script/test0.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})