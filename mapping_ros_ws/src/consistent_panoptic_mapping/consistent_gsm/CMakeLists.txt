cmake_minimum_required(VERSION 3.6)
project(consistent_gsm)

add_definitions(-std=c++14)

set(CATKIN_ENABLE_TESTING OFF)

find_package(catkin_simple REQUIRED)
catkin_simple()
find_package(catkin REQUIRED COMPONENTS
global_segment_map
gsm_node
opencv3_catkin
# pybind11_catkin
)

## Declare the catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
  ${PROJECT_NAME}
  CATKIN_DEPENDS
    global_segment_map
    gsm_node
    opencv3_catkin
    voxblox
    voxblox_ros
    # pybind11_catkin
)

##########
# Pybind #
##########
include_directories(cvnp)
find_package(pybind11 REQUIRED)

##########
# this project   #
##########
include_directories(include)
pybind11_add_module(${PROJECT_NAME} src/global_segment_map_py.cpp)

##########
# omp   #
##########
if(APPLE)
  set(LLVM_PATH "/usr/local/opt/llvm")
  find_package(LLVM HINTS ${LLVM_PATH}/lib/cmake/llvm)
  if (LLVM_FOUND)
    message(STATUS "Building with LLVM clang and OpenMP support.")
    set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
    set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -L${LLVM_PATH}/lib -Wl,-rpath,${LLVM_PATH}/lib")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -L${LLVM_PATH}/lib -Wl,-rpath,${LLVM_PATH}/lib")
  else()
    message(WARNING "LLVM not found: Building without OpenMP support.")
  endif()
else()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: Adding OpenMP support.")
    target_compile_options(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_LIBRARIES})
  else()
    message(WARNING "OpenMP not found: Building without OpenMP support.")
  endif()
endif()


##########
# cvnp   #
##########
add_subdirectory(cvnp)

##########
# this project   #
##########
cs_add_library(${PROJECT_NAME}_LIB
  # src/segment_graph.cpp //moved to global_segment_map
  src/SettingsSemantic.cpp
  src/segment_confidence.cpp
  src/label_tsdf_confidence_integrator.cpp
  src/consistent_gsm_mapper.cpp
  src/utils/pcl_semantic_visualizers.cpp
)
target_link_libraries(${PROJECT_NAME}_LIB ${OpenMP_LIBS})
target_compile_options(${PROJECT_NAME}_LIB PRIVATE ${OpenMP_FLAGS})

target_link_libraries(${PROJECT_NAME} PRIVATE cvnp ${catkin_LIBRARIES} ${PROJECT_NAME}_LIB)
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}/${PROJECT_NAME}
)


set(ENV{PYTHONPATH} ENV{PYTHONPATH} "${CATKIN_PACKAGE_LIB_DESTINATION}")
# list(APPEND PYTHONPATH "${CATKIN_PACKAGE_LIB_DESTINATION}") 
MESSAGE(STATUS "PYTHONPATH: ${PYTHONPATH}")
MESSAGE(STATUS "CATKIN_PACKAGE_LIB_DESTINATION: ${CATKIN_PACKAGE_LIB_DESTINATION}")
# catkin_python_setup(PROGRAMS ${PROJECT_NAME})

cs_install()
cs_export()

