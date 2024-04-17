

cmake_minimum_required(VERSION 3.12)
project("tortoise")


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(tortoise main.cpp common.cpp)

set(GGML_CUBLAS ON)
set(GGML_USE_CUBLAS ON)

add_definitions(-DGGML_USE_CUBLAS)

add_subdirectory(ggml)

target_link_libraries(tortoise PUBLIC ggml)
target_include_directories(tortoise PUBLIC . )
target_compile_features(tortoise PUBLIC cxx_std_11)