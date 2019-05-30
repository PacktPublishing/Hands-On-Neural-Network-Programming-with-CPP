####################################################################################################################
###
### Project's CMake configuration file
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
### |__src
### |   |__main.cpp
### |
### |__CMakeLists.txt
###
### - The src folder has just .cpp files
### - In the root of the structure lies this CMakeLists.txt file.
###
####################################################################################################################

cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME mlp_core)
project(${PROJECT_NAME} CXX)

# set default build type to release
if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

add_executable(linear-example ${PROJECT_SOURCE_DIR}/src/classification.cpp)
target_compile_options(linear-example PRIVATE -Wall -Wextra -pedantic)

add_executable(feedforwading-example ${PROJECT_SOURCE_DIR}/src/feedforwading.cpp)
target_compile_options(feedforwading-example PRIVATE -Wall -Wextra -pedantic)