find_package(OpenCV REQUIRED)

set(common_libs
    CUDA::cuda_driver
    zenohcxx::zenohpico
    opencv_core
    opencv_imgproc
    opencv_videoio
    node)

add_executable(video_reader)
target_sources(video_reader
    PRIVATE
    video_reader.cpp)
target_include_directories(video_reader
    PRIVATE
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(video_reader
    PRIVATE
    ${common_libs})

add_executable(blur)
target_sources(blur
    PRIVATE
    blur.cu)
target_include_directories(blur
    PRIVATE
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(blur
    PRIVATE
    ${common_libs})