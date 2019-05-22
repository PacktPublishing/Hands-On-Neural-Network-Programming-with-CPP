####################################################################################################################
###
### Project's CMake configuration file
###
### This file is used for build the project. It calls some other files to clone/download/configure third part libraries.
### The project is set to C++17 standard. You can check if your compiler has support for C++17 in this link: 
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
### to run the project just call
### $ ./backprop_example
###
### You can export the CXX variable if you decide to use a specific compiler. For example, on macos, to use clang++ 7.0.1 you
### must to perform the following command BEFORE call cmake:
###
### $ export CXX=/usr/local/Cellar/llvm/7.0.1/bin/clang++
###
### After that, call cmake with the flag -DCMAKE_PREFIX_PATH=/your/path:
###
### cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/usr/local/Cellar/llvm/7.0.1/ ..
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
### |
### |__libs
### |   |__ .
### |
### |__src
### |   |__main.cpp
### |
### |__CMakeLists.txt
###
### - The libs folder is where the third party library's CMake configuration files are located.
### - The src folder has just one main.cpp file.
### - Finally, in the root of the structure lies the CMakeLists.txt folder.
###
####################################################################################################################

cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME gradient_descent)
project(${PROJECT_NAME} CXX)

# set the default build type to release
if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# download header-only libraries

include(libs/eigen/install.txt)

include_directories(include)

add_executable(gradient_descent ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_compile_options(gradient_descent PRIVATE -Wall -Wextra -pedantic)