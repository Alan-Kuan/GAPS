find_package(OpenCV REQUIRED)

set(common_libs
    zenohcxx::zenohpico
    opencv_core
    opencv_imgproc
    opencv_videoio
    node)

shoz_add_target(pub_preproc
    IS_EXE
    SRCS pub_preproc.cpp
    INCS ${OpenCV_INCLUDE_DIRS}
    LIBS ${common_libs})

shoz_add_target(sub_blur
    IS_EXE
    SRCS sub_blur.cpp blur.cu
    INCS ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/..
    LIBS ${common_libs} CUDA::cudart helpers)