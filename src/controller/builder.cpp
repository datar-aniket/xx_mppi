#include "xx_mppi/controller/builder.hpp"

#include <memory>
#include <utility>

#include "xx_mppi/controller/config.hpp"
#include "xx_mppi/reference/raceline.hpp"

namespace xxcar::mppi {

std::unique_ptr<MppiController> MppiControllerBuilder::FromConfigDirectory(
  const std::string & config_directory, const std::string & vehicle_config_file)
{
  auto config = LoadControllerConfig(config_directory, vehicle_config_file);
  auto raceline = Raceline::LoadCsv(config.raceline_path);
  return std::make_unique<MppiController>(std::move(config), std::move(raceline));
}

}  // namespace xxcar::mppi
