cmake_minimum_required(VERSION 3.20)

project(vio)

find_package(OpenCV REQUIRED)
find_package(GSL REQUIRED)

add_executable(${CMAKE_PROJECT_NAME}
    "src/main.cpp"
    "src/Dataset.cpp"
    "src/msckf.c"
    "src/Client.cpp"
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    "include"
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    -O3
    -Wall
    -Wextra
    -Wpedantic
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    GSL::gsl
)
