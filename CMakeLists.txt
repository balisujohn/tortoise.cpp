cmake_minimum_required(VERSION 3.12)
project("tortoise")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DEBUG "Debug mode" OFF)
option(GGML_CUDA "cuda mode" OFF)

if(DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O0")
    set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -g3 -O0")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -g3 -O0")
    set(GGML_DEBUG ON CACHE BOOL "Debug mode")
endif()

if (GGML_CUDA)
    add_definitions(-DGGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_definitions(-DGGML_USE_METAL)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed Metal library")
endif()

add_subdirectory(ggml)
add_executable(tortoise main.cpp common.cpp)
target_link_libraries(tortoise PUBLIC ggml)
target_include_directories(tortoise PUBLIC . )
target_compile_features(tortoise PUBLIC cxx_std_11)
