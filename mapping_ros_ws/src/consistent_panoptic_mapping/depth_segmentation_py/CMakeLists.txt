cmake_minimum_required(VERSION 2.8.3)
project(depth_segmentation_py)

add_definitions(-std=c++14)

find_package(catkin_simple REQUIRED)
catkin_simple()
find_package(catkin REQUIRED COMPONENTS
depth_segmentation
opencv3_catkin
# pybind11_catkin
)
if(depth_segmentation_FOUND_CATKIN_PROJECT)
    MESSAGE(STATUS "Found depth_segmentation")
endif(depth_segmentation_FOUND_CATKIN_PROJECT)
# find_package(depth_segmentation REQUIRED)

## Declare the catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
  ${PROJECT_NAME}
  CATKIN_DEPENDS
    depth_segmentation
    opencv3_catkin
    # pybind11_catkin
)

include_directories(include)
link_directories(depth_segmentation)
##########
# cvnp   #
##########
add_subdirectory(cvnp)

##########
# Pybind #
##########
include_directories(cvnp)
find_package(pybind11 REQUIRED)

pybind11_add_module(${PROJECT_NAME} src/depth_segmentation_py.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE cvnp ${catkin_LIBRARIES})
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}/${PROJECT_NAME}
)
# install(
#   TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}/${PROJECT_NAME}
# )

set(ENV{PYTHONPATH} ENV{PYTHONPATH} "${CATKIN_PACKAGE_LIB_DESTINATION}")
# list(APPEND PYTHONPATH "${CATKIN_PACKAGE_LIB_DESTINATION}") 
MESSAGE(STATUS "PYTHONPATH: ${PYTHONPATH}")
MESSAGE(STATUS "CATKIN_PACKAGE_LIB_DESTINATION: ${CATKIN_PACKAGE_LIB_DESTINATION}")
# catkin_python_setup(PROGRAMS ${PROJECT_NAME})