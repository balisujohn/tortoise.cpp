

cmake_minimum_required(VERSION 3.12)
project("tortoise")


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(tortoise main.cpp common.cpp)

option(GGML_CUBLAS "cuda mode" OFF)

if (GGML_CUBLAS)
    add_definitions(-DGGML_USE_CUBLAS)
endif()

add_subdirectory(ggml)

target_link_libraries(tortoise PUBLIC ggml)
target_include_directories(tortoise PUBLIC . )
target_compile_features(tortoise PUBLIC cxx_std_11)