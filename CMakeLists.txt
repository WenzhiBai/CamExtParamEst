#cmake版本最低要求
cmake_minimum_required(VERSION 2.8)

#项目名称
project(CamExtParamEst)

#(可选)如果没有指定cmake编译模式，就选择Release模式,必须写成三行
IF(NOT CMAKE_BUILD_TYPE)
SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#添加c++ 11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

#设置生成的so动态库最后输出的路径
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

#将产生的target文件输出到指定路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

#添加头文件目录,相当于指定gcc的-I参数
#include_directories(${PROJECT_SOURCE_DIR}/include)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3_DIR: ${Eigen3_DIR}")

set(ceres_DIR "/usr/local/lib/cmake/Ceres")
find_package(Ceres REQUIRED)
message(STATUS "ceres_DIR: ${ceres_DIR}")

aux_source_directory(src DIR_SRCS)
add_executable(main ${DIR_SRCS})

target_link_libraries(main Eigen3::Eigen ceres)
