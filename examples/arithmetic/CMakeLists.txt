shoz_add_target(run
    IS_EXE
    SRCS src/run.cu
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/include
    LIBS zenohcxx::zenohpico node vector_arithmetic)

shoz_add_target(vector_arithmetic
    SRCS src/vector_arithmetic.cu
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/include)