add_executable(${EXE})
target_sources(${EXE} PRIVATE main.cpp)
target_include_directories(${EXE} PRIVATE ../include)
target_link_libraries(${EXE} PRIVATE comm)

add_library(comm STATIC)
target_sources(comm PRIVATE pub.cpp sub.cpp)
target_include_directories(comm PRIVATE ../include)
target_link_libraries(comm
    PUBLIC zenohcxx::zenohc::lib
    PRIVATE work)

add_library(work STATIC)
target_sources(work PRIVATE work.cu)
target_include_directories(work PRIVATE ../include)