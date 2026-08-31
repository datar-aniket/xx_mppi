#pragma once

#include <memory>
#include <string>

#include "xx_mppi/controller/mppi_controller.hpp"

namespace xxcar::mppi {

class MppiControllerBuilder {
 public:
  static std::unique_ptr<MppiController> FromConfigDirectory(
    const std::string & config_directory);
};

}  // namespace xxcar::mppi
