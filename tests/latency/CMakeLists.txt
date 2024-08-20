shoz_add_target(run_test
    IS_EXE
    SRCS run_test.cpp
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/..
    LIBS CUDA::cuda_driver zenohcxx::zenohpico node)