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

Rows must have finite numeric values and strictly increasing `s`. A loop should
include a final point near the first position with the final lap-length `s`, as
in EPIC exports. Bounds use `e_min < 0 < e_max` and are static; this package has
no runtime map or perception update path.

