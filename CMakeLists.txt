cmake_minimum_required(VERSION 3.16)
project(CNN_PureCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(cnn_mnist
    main.cpp
    mnist_reader.cpp
)

target_include_directories(cnn_mnist PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optimization flags
if(MSVC)
    target_compile_options(cnn_mnist PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(cnn_mnist PRIVATE -O3 -march=native -mtune=native)
endif()

target_link_libraries(cnn_mnist PRIVATE)
