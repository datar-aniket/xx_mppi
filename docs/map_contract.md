# Static raceline CSV contract

The loader accepts the EPIC CSV layout. Required columns and aliases are:

| Meaning | Accepted headers | Units |
|---|---|---|
| path evolution | `s` | m |
| curvature | `k`, `kappa`, `curvature` | 1/m |
| ENU position | `E`/`N` (also `east`/`north`, `x`/`y`) | m |
| EPIC heading | `phi`, `psi`, `heading` | rad |
| speed profile | `V`, `speed`, `v` | m/s |
| track bounds | `e_min`, `e_max` | m |

Optional feed-forward/body columns are `r`, `beta`, `delta_cmd`, torque
(`wheel_torque_cmd`, `torque_cmd`, or EPIC `engine_torque_cmd`), and driven-wheel
speed (`rear_wr`, `driven_wheel_speed`, or `motor_speed_mps`). Missing optional
values default to zero, except driven-wheel speed defaults to `V`.

`phi` is measured counter-clockwise from north, so the centreline tangent is
`(-sin(phi), cos(phi))` and `phi = yaw_enu - pi/2`. `k` must equal `dphi/ds`,
positive for a left turn. Positive `e` is the left normal of that tangent, so
`e_max` is the left boundary and `e_min` the right one. Every map under
`~/map` produced by `raceline_generator` follows this convention; the
`Frames.CsvHeadingMatchesSurveyedTangent` test checks a surveyed loop against
its own geometry.

Rows must have finite numeric values and strictly increasing `s`. A loop should
include a final point near the first position with the final lap-length `s`, as
in EPIC exports. Bounds use `e_min < 0 < e_max` and are static; this package has
no runtime map or perception update path. If a closed sampled loop omits the
duplicate endpoint, the loader appends the first sample virtually at
`last_s + closing_distance` so Frenet wrapping remains spatially continuous.

The runtime `model.yaml` may select a map directory through the lowercase
`current_map` environment variable:

```bash
export current_map="$HOME/map/last_run_sept1"
ros2 launch xx_mppi mppi.launch.py
```

With `raceline_path: "${current_map}"`, the directory basename is used to derive
`$current_map/<basename>_frenet_map.csv`; the example therefore loads
`/home/jetson/map/last_run_sept1/last_run_sept1_frenet_map.csv`. Direct CSV
paths, relative paths, `$VAR`/`${VAR}`, and `~` are also supported. Startup
fails with a specific error when a referenced variable, directory, or derived
CSV is missing.

`config/raceline.csv` is a sample 10 m diameter circular loop for bench and ROS
pipeline tests. Its companion `config/map.yaml` records the ENU geometry and a
test start pose, but is not loaded by the controller; the CSV centerline and
`e_min`/`e_max` columns are the authoritative runtime map. Replace the sample
with surveyed track data before autonomous operation.
