include_directories(include)

add_executable(cross_process)
target_sources(cross_process
    PRIVATE
    src/cross_process.cu)
target_link_libraries(cross_process
    PRIVATE
    CUDA::cuda_driver
    zenohcxx::zenohpico
    vector_arithmetic
    node
    error)

add_library(vector_arithmetic STATIC)
target_sources(vector_arithmetic
    PRIVATE
    src/vector_arithmetic.cu)