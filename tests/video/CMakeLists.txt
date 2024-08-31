find_package(OpenCV REQUIRED)

set(common_libs
    zenohcxx::zenohpico
    opencv_core
    opencv_imgproc
    opencv_videoio
    node)

shoz_add_target(video_reader
    IS_EXE
    SRCS video_reader.cpp
    INCS ${OpenCV_INCLUDE_DIRS}
    LIBS ${common_libs})

shoz_add_target(blur
    IS_EXE
    SRCS blur.cu
    INCS ${OpenCV_INCLUDE_DIRS}
    LIBS ${common_libs})