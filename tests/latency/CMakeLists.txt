add_executable(test0)
target_sources(test0
    PRIVATE
    test0.cpp)
target_include_directories(test0
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test0
    PRIVATE
    CUDA::cuda_driver
    zenohcxx::zenohpico
    node
)

add_executable(test1)
target_sources(test1
    PRIVATE
    test1.cpp)
target_include_directories(test1
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test1
    PRIVATE
    CUDA::cuda_driver
    zenohcxx::zenohpico
    node
)

set(SCRIPT_DIR "${CMAKE_BINARY_DIR}/tests/latency")
file(MAKE_DIRECTORY ${SCRIPT_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/script/test0.sh DESTINATION ${SCRIPT_DIR})