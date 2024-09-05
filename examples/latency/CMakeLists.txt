shoi_add_target(run_test
    IS_EXE
    SRCS run_test.cpp
    INCS ${CMAKE_CURRENT_SOURCE_DIR}/..
    LIBS CUDA::cudart iceoryx_posh::iceoryx_posh node helpers tag_init)

shoi_add_target(tag_init
    SRCS init.cu)
