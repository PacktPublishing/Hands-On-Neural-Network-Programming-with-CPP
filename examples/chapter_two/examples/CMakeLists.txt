####################################################################################################################
###
### Project's CMake configuration file
###
### This file is used for build the project. It calls some other files to clone/download/configure third part libraries.
### The project is set to C++11 standard. You can check if your compiler has support for C++11 in this link: 
### https://en.cppreference.com/w/cpp/compiler_support
###
### BUILD INSTRUCTIONS
###
### To build the project, no administrative privilegies are required. Assuming that you are in the root folder, just run:
###
### $ 
### $ mkdir build
### $ cd build
### $ cmake ..
### $ make
### $
###
### By default, this file configure the build for Release type. Hence, if you want to build with Debug information, 
### include the following property: -DCMAKE_BUILD_TYPE=Debug . For example, if you want to build the project with Debug
### information and tests, run:
###
### $ cmake -DCMAKE_BUILD_TYPE=Debug ..
###
### Whenever you add new *.cpp or *.hpp files, make sure you clean the build folder and run cmake again. To clean the build 
### folder just delete all of it contents. In a *-nix based system like linux or OSX and assuming that you are in the root folder,
### just type:
###
### $
### $ cd build
### $ rm -rf *
### $ cmake ..
### $ make
###
### DISCLAIMER: double check if you are actually in the build folder before run 'rm -rf *'
###
### PROJECT FOLDER STRUCTURE
###
### This file assumes the following folder structure
###
### .
### |
### |__data
### |   |__ .
### |
### |__include
### |   |__ .
### |
### |__libs
### |   |__ .
### |
### |__src
### |   |__lib
### |   |   |__ .
### |   |
### |   |__main.cpp
### |
### |__CMakeLists.txt
###
### - The data folder Files with data for testing purpose are found in the data folder.
### - The library headers are stored in the include folder.
### - The libs folder is where the third party library's CMake configuration files are located.
### - The src folder has the .cpp files and the lib nested folder where the library's *.cpp files are stored.
### - Finally, in the root of the structure lies this CMakeLists.txt file.
###
####################################################################################################################

cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME chapter2_examples)
project(${PROJECT_NAME} CXX)

# set default build type to release
if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# download header-only libraries

include(libs/eigen/install.txt)
include(libs/fast-cpp-csv-parser/install.txt)

include_directories(include)

# be careful when using file globbing!
file(GLOB SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp")
add_library(${PROJECT_NAME}_lib ${SOURCES_LIB})

add_executable(activation_usage ${PROJECT_SOURCE_DIR}/src/activation_usage.cpp)
target_compile_options(activation_usage PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(activation_usage ${PROJECT_NAME}_lib)

add_executable(mse_eval ${PROJECT_SOURCE_DIR}/src/mse_eval.cpp)
target_compile_options(mse_eval PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(mse_eval ${PROJECT_NAME}_lib)

add_executable(softmax_example ${PROJECT_SOURCE_DIR}/src/softmax_example.cpp)
target_compile_options(softmax_example PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(softmax_example ${PROJECT_NAME}_lib)

add_executable(example_2 ${PROJECT_SOURCE_DIR}/src/example_2.cpp)
target_compile_options(example_2 PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(example_2 ${PROJECT_NAME}_lib)
