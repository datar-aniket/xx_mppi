#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "xx_mppi/ros/mppi_node.hpp"

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<xxcar::mppi::MppiNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 2U);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("xx_mppi_node"), "Startup failed: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
