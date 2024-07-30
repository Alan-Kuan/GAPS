add_executable(cross_process)
target_sources(cross_process
    PRIVATE
    src/cross_process.cu)
target_include_directories(cross_process
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cross_process
    PRIVATE
    CUDA::cuda_driver
    zenohcxx::zenohpico
    vector_arithmetic
    node)

add_library(vector_arithmetic STATIC)
target_sources(vector_arithmetic
    PRIVATE
    src/vector_arithmetic.cu)
target_include_directories(vector_arithmetic
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)