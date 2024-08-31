shoi_add_target(run
    IS_EXE
    SRCS src/run.cpp
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/include
    LIBS CUDA::cudart iceoryx_posh::iceoryx_posh node vector_arithmetic)

shoi_add_target(vector_arithmetic
    SRCS src/vector_arithmetic.cu
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/include)