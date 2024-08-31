shoz_add_target(run
    IS_EXE
    SRCS src/run.cpp
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/include
    LIBS CUDA::cudart zenohcxx::zenohpico node vector_arithmetic helpers)

shoz_add_target(vector_arithmetic
    SRCS src/vector_arithmetic.cu
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/include)