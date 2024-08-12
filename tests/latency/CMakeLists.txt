shoz_add_target(test0
    IS_EXE
    SRCS test0.cpp
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/..
    LIBS CUDA::cuda_driver zenohcxx::zenohpico node)

shoz_add_target(test1
    IS_EXE
    SRCS test1.cpp
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/..
    LIBS CUDA::cuda_driver zenohcxx::zenohpico node)

set(SCRIPT_DIR "${CMAKE_BINARY_DIR}/tests/latency")
file(MAKE_DIRECTORY ${SCRIPT_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/script/test0.sh DESTINATION ${SCRIPT_DIR})