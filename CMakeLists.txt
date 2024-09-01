cmake_minimum_required(VERSION 3.25.2)
project(shoi LANGUAGES CXX CUDA)

option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TORCH_SUPPORT "Build a Python module with PyTorch support" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(iceoryx_posh CONFIG REQUIRED)
if(BUILD_TORCH_SUPPORT)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

    execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
endif()

# Utility Function
function(shoi_add_target targetName)
    set(flags IS_EXE PIC)
    set(listArgs SRCS INCS LIBS)
    cmake_parse_arguments(ARG "${flags}" "" "${listArgs}" ${ARGN})

    if (NOT ARG_SRCS)
        message(FATAL_ERROR "[shoi_add_target]: missing required argument SRCS")
    endif()

    if (ARG_IS_EXE)
        add_executable(${targetName})
    else()
        add_library(${targetName} STATIC)
    endif()

    target_sources(${targetName} PRIVATE ${ARG_SRCS})

    if (ARG_INCS)
        target_include_directories(${targetName} PRIVATE ${ARG_INCS})
    endif()
    if (ARG_LIBS)
        target_link_libraries(${targetName} PRIVATE ${ARG_LIBS})
    endif()

    if (ARG_PIC)
        set_target_properties(${targetName} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endfunction()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
