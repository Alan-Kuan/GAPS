shoz_add_target(cross_process
    IS_EXE
    SRCS src/cross_process.cu
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/include
    LIBS zenohcxx::zenohpico node vector_arithmetic)

shoz_add_target(vector_arithmetic
    SRCS src/vector_arithmetic.cu
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/include)