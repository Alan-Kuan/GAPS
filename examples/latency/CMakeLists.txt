prj_add_target(run_test
    IS_EXE
    SRCS run_test.cpp
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/..
    LIBS zenohcxx::zenohpico node utils init)

prj_add_target(init
    SRCS init.cu)