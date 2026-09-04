cmake_minimum_required(VERSION 3.18)
project(xx_mppi LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(XX_MPPI_ENABLE_CUDA "Build the CUDA MPPI backend" ON)
option(XX_MPPI_ENABLE_TENSORRT "Build the TensorRT learned-model backend" ON)
set(XX_MPPI_CUDA_ARCHITECTURES "87" CACHE STRING
  "CUDA architectures; 87 is the Jetson Orin GPU")

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(xxcar_msgs REQUIRED)

add_library(xx_mppi_core SHARED
  src/controller/builder.cpp
  src/controller/config.cpp
  src/controller/mppi_controller.cpp
  src/costs/cost_evaluator.cpp
  src/costs/obstacle_cost.cpp
  src/dynamics/rollout.cpp
  src/obstacles/laser_deskew.cpp
  src/obstacles/pose_history.cpp
  src/obstacles/signed_distance_field.cpp
  src/reference/raceline.cpp
)
target_include_directories(xx_mppi_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(xx_mppi_core PUBLIC yaml-cpp)
target_compile_options(xx_mppi_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>)

add_library(xx_mppi_ros SHARED
  src/ros/direct_control_message.cpp
  src/ros/ekf_state_adapter.cpp
  src/ros/mppi_node.cpp
  src/ros/runtime.cpp
  src/ros/runtime_config.cpp
  src/ros/trajectory_message.cpp
  src/ros/visualization.cpp)
target_include_directories(xx_mppi_ros PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(xx_mppi_ros xx_mppi_core)
ament_target_dependencies(
  xx_mppi_ros ament_index_cpp geometry_msgs nav_msgs rclcpp
  sensor_msgs std_msgs tf2 tf2_ros visualization_msgs xxcar_msgs)
target_compile_options(xx_mppi_ros PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

add_executable(xx_mppi_node src/ros/main.cpp)
target_link_libraries(xx_mppi_node xx_mppi_ros)
ament_target_dependencies(xx_mppi_node rclcpp)
target_compile_options(xx_mppi_node PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

add_executable(xxcar_benchmark_sdf tools/benchmark_obstacle_sdf.cpp)
target_link_libraries(xxcar_benchmark_sdf xx_mppi_core)
target_compile_options(xxcar_benchmark_sdf PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion)

set(XX_MPPI_HAS_CUDA OFF)
if(XX_MPPI_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(XX_MPPI_HAS_CUDA ON)
    target_sources(xx_mppi_core PRIVATE src/cuda/cuda_mppi.cu)
    target_link_libraries(xx_mppi_core PRIVATE CUDA::cudart CUDA::curand)
    target_compile_definitions(xx_mppi_core PUBLIC XX_MPPI_HAS_CUDA=1)
    set_target_properties(xx_mppi_core PROPERTIES
      CUDA_ARCHITECTURES "${XX_MPPI_CUDA_ARCHITECTURES}"
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON)
    target_compile_options(xx_mppi_core PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math>)

    add_executable(xxcar_benchmark_mppi tools/benchmark_cuda_mppi.cpp)
    target_link_libraries(xxcar_benchmark_mppi xx_mppi_core)
    target_compile_options(xxcar_benchmark_mppi PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion)
  else()
    message(WARNING "CUDA compiler not found; building the CPU numerical oracle only")
    target_sources(xx_mppi_core PRIVATE src/controller/cuda_mppi_stub.cpp)
  endif()
else()
  target_sources(xx_mppi_core PRIVATE src/controller/cuda_mppi_stub.cpp)
endif()

set(XX_MPPI_HAS_TENSORRT OFF)
if(XX_MPPI_ENABLE_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/local/include)
  find_library(TENSORRT_NVINFER nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/lib)
  find_library(TENSORRT_NVONNXPARSER nvonnxparser
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/lib)
  if(XX_MPPI_HAS_CUDA AND TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER AND TENSORRT_NVONNXPARSER)
    set(XX_MPPI_HAS_TENSORRT ON)
    target_compile_definitions(xx_mppi_core PUBLIC XX_MPPI_HAS_TENSORRT=1)
    target_sources(xx_mppi_core PRIVATE src/dynamics/tensorrt_model.cpp)
    target_include_directories(xx_mppi_core PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(xx_mppi_core PRIVATE ${TENSORRT_NVINFER})

    add_executable(xxcar_build_trt_engine tools/build_tensorrt_engine.cpp)
    target_include_directories(xxcar_build_trt_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(xxcar_build_trt_engine PRIVATE
      CUDA::cudart ${TENSORRT_NVINFER} ${TENSORRT_NVONNXPARSER})
    target_compile_options(xxcar_build_trt_engine PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion)
  else()
    message(STATUS "TensorRT not found; learned-model backend disabled on this host")
    target_sources(xx_mppi_core PRIVATE src/dynamics/tensorrt_model_stub.cpp)
  endif()
else()
  target_sources(xx_mppi_core PRIVATE src/dynamics/tensorrt_model_stub.cpp)
endif()

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch docs DESTINATION share/${PROJECT_NAME})
install(TARGETS xx_mppi_core xx_mppi_ros xx_mppi_node xxcar_benchmark_sdf
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
if(XX_MPPI_HAS_CUDA)
  install(TARGETS xxcar_benchmark_mppi RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()
if(XX_MPPI_HAS_TENSORRT)
  install(TARGETS xxcar_build_trt_engine RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()
install(PROGRAMS
  tools/export_pytorch_dynamics.py
  tools/publish_sample_ekf_state.py
  tools/smoke_test_ros_pipeline.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_raceline test/test_raceline.cpp)
  target_link_libraries(test_raceline xx_mppi_core)
  target_compile_definitions(test_raceline PRIVATE
    XX_MPPI_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  ament_add_gtest(test_frames test/test_frames.cpp)
  target_link_libraries(test_frames xx_mppi_core)
  target_compile_definitions(test_frames PRIVATE
    XX_MPPI_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  ament_add_gtest(test_dynamics_costs test/test_dynamics_costs.cpp)
  target_link_libraries(test_dynamics_costs xx_mppi_core)
  target_compile_definitions(test_dynamics_costs PRIVATE
    XX_MPPI_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  ament_add_gtest(test_config_message test/test_config_message.cpp)
  target_link_libraries(test_config_message xx_mppi_ros)
  target_compile_definitions(test_config_message PRIVATE
    XX_MPPI_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")
  set_tests_properties(test_config_message PROPERTIES
    ENVIRONMENT "current_map=${CMAKE_CURRENT_SOURCE_DIR}/test/current_map")

  ament_add_gtest(test_ekf_state_adapter test/test_ekf_state_adapter.cpp)
  target_link_libraries(test_ekf_state_adapter xx_mppi_ros)
  target_compile_definitions(test_ekf_state_adapter PRIVATE
    XX_MPPI_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  ament_add_gtest(test_visualization test/test_visualization.cpp)
  target_link_libraries(test_visualization xx_mppi_ros)
  target_compile_definitions(test_visualization PRIVATE
    XX_MPPI_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  ament_add_gtest(test_obstacles test/test_obstacles.cpp)
  target_link_libraries(test_obstacles xx_mppi_core)
endif()

ament_export_include_directories(include)
ament_export_libraries(xx_mppi_core xx_mppi_ros)
ament_export_dependencies(
  ament_index_cpp geometry_msgs nav_msgs rclcpp std_msgs
  sensor_msgs tf2 tf2_ros visualization_msgs xxcar_msgs yaml-cpp)
ament_package()
