cmake_minimum_required(VERSION 3.6)
project(global_segment_map)

add_definitions(-std=c++14)
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_definitions(-fext-numeric-literals)
endif()
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
find_package(opencv3_catkin REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++fs")


include_directories(include ${catkin_INCLUDE_DIRS} )
message(STATUS "OPENCV3_CATKIN_INCLUDE: ${OPENCV3_CATKIN_INCLUDE}")
message(STATUS "catkin_INCLUDE_DIRS: ${catkin_INCLUDE_DIRS}")
cs_add_library(${PROJECT_NAME}
  src/segment_graph.cpp
  src/label_block_serialization.cc
  src/semantic_instance_label_fusion.cc
  src/label_merge_integrator.cc
  src/icp_utils.cc
  src/label_tsdf_integrator.cc
  src/label_tsdf_map.cc
  src/meshing/label_tsdf_mesh_integrator.cc
  src/meshing/label_color_map.cc
  src/meshing/instance_color_map.cc
  src/meshing/semantic_color_map.cc
  src/meshing/open_sem_color_map.cc
  src/segment.cc
  src/utils/visualizer.cc
)

cs_install()
cs_export()
