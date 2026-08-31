#include "xx_mppi/controller/builder.hpp"

#include <memory>
#include <utility>

#include "xx_mppi/controller/config.hpp"
#include "xx_mppi/reference/raceline.hpp"

namespace xxcar::mppi {

std::unique_ptr<MppiController> MppiControllerBuilder::FromConfigDirectory(
  const std::string & config_directory)
{
  auto config = LoadControllerConfig(config_directory);
  auto raceline = Raceline::LoadCsv(config.raceline_path);
  return std::make_unique<MppiController>(std::move(config), std::move(raceline));
}

}  // namespace xxcar::mppi
