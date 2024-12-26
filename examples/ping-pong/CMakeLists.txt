prj_add_target(ping-pong
    IS_EXE
    SRCS main.cu
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/..
    LIBS zenohcxx::zenohpico node rand_init)

prj_add_target(rand_init
    SRCS rand_init.cu)