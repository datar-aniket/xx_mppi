"""Numpy replica of DynamicBicycleFiala4ws (include/xx_mppi/dynamics/models/dynamic_bicycle_fiala_4ws.hpp)
with longitudinal load transfer, vectorized over rollouts.

Load transfer moves dFz = m h ax_lt / L from the front axle to the rear, where
ax_lt is the body-frame acceleration of the tire forces. Two ways to close the
loop are provided: carry ax_lt from the previous substep (optionally through a
first-order lag), or iterate inside one derivative evaluation (fixed point).
"""

import numpy as np

GRAVITY = 9.81  # the C++ model's constant


class Params(dict):
    __getattr__ = dict.get


# carxx vehicle.yaml as used for the IROS runs (2026-09-27), load transfer off.
CARXX_IROS = Params(m=2.465, Iz=0.0305, a=0.13, b=0.13, Cf=150.0, Cr=150.0, muf=0.35, mur=0.35, R=0.03,
                    J=0.0015, bias=0.5, locked=True, RR=1.8,
                    h=0.0,       # load-transfer height [m]
                    tau_lt=0.0,  # lag of ax_lt [s]; 0 takes the previous substep's value
                    c_exp=0.0)   # cornering stiffness scales with (Fz / Fz_static)^c_exp


def params_from_vehicle_yaml(path):
    import yaml
    with open(path) as f:
        v = yaml.safe_load(f)['vehicle']
    g, d, t = v['geometry'], v['drivetrain'], v['tires']
    return Params(CARXX_IROS, m=v['mass_kg'], Iz=v['yaw_inertia_kgm2'], a=g['cg_to_front_m'], b=g['cg_to_rear_m'],
                  Cf=t['front_cornering_stiffness_nprad'], Cr=t['rear_cornering_stiffness_nprad'],
                  muf=t['front_friction_coefficient'], mur=t['rear_friction_coefficient'], R=t['wheel_radius_m'],
                  J=d['driven_wheel_inertia_kgm2'], bias=d.get('front_brake_bias', 0.0),
                  locked=bool(d.get('locked_awd', False)), RR=d.get('rolling_resistance_n', 0.0),
                  h=g.get('load_transfer_height_m', 0.0))


def fiala_combined(alpha, kappa, C, mu, Fz):
    """FialaCombinedSlip in tires.hpp. Returns (lateral, longitudinal)."""
    amax = 0.5 * np.pi - 1e-2
    alpha = np.clip(alpha, -amax, amax)
    fmax = np.maximum(mu * Fz, 1e-6)
    C = np.maximum(C, 1e-6)
    sig_sat = np.arctan(3.0 * fmax / C)
    tan_a = np.tan(alpha)
    sig = np.maximum(np.sqrt(tan_a * tan_a + kappa * kappa + 1e-12), 1e-10)
    unsaturated = C * sig - C * C / (3.0 * fmax) * sig**2 + C**3 / (27.0 * fmax**2) * sig**3
    mag = np.where(np.abs(sig) > sig_sat, fmax, unsaturated)
    return -mag * tan_a / sig, mag * kappa / sig


def derivative(x, u, p, ax_lt):
    """x (N,4) [yaw rate, speed, sideslip, driven wheel speed]; u (N,3) [front steer, torque, rear steer].

    Returns (dx, body_ax, body_ay, (Fz_front, Fz_rear)).
    """
    if not p.locked:
        raise NotImplementedError('only the locked-AWD driveline (carxx) is replicated')
    r, v_in, beta_in, w = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    d, tau, dr = u[:, 0], u[:, 1], u[:, 2]
    m, Iz, a, b = p.m, p.Iz, p.a, p.b
    L = a + b
    fzf0, fzr0 = m * GRAVITY * b / L, m * GRAVITY * a / L
    # Clamped so neither axle goes negative; the total stays m g.
    dfz = np.clip(m * p.h * ax_lt / L, -fzr0, fzf0)
    fzf, fzr = fzf0 - dfz, fzr0 + dfz
    Cf = p.Cf * (np.maximum(fzf, 1e-6) / fzf0) ** p.c_exp
    Cr = p.Cr * (np.maximum(fzr, 1e-6) / fzr0) ** p.c_exp
    v = np.copysign(np.maximum(np.abs(v_in), 1.0), v_in)
    beta = np.where(np.abs(v_in) > 1e-3, beta_in, 0.0)
    cb, sb = np.cos(beta), np.sin(beta)
    cd, sd = np.cos(d), np.sin(d)
    cdr, sdr = np.cos(dr), np.sin(dr)
    cdb, sdb = np.cos(d - beta), np.sin(d - beta)
    cdrb, sdrb = np.cos(dr - beta), np.sin(dr - beta)
    vx, vy, true_vx = v * cb, v * sb, v_in * cb
    alpha_f = np.arctan2(vy + a * r, vx) - d
    alpha_r = np.arctan2(vy - b * r, vx) - dr
    front_long = cd * true_vx + sd * (vy + a * r)
    kappa_f = (w - front_long) / np.maximum(np.abs(front_long), 1.0)
    rear_long = cdr * true_vx + sdr * (vy - b * r)
    kappa_r = (w - rear_long) / np.maximum(np.abs(rear_long), 1.0)
    fyf, fxf = fiala_combined(alpha_f, kappa_f, Cf, p.muf, fzf)
    fyr, fxr = fiala_combined(alpha_r, kappa_r, Cr, p.mur, fzr)
    yaw_acc = (a * fxf * sd + a * fyf * cd - b * fxr * sdr - b * fyr * cdr) / Iz
    rolling = p.RR * np.tanh(v_in / 0.1)
    v_dot = (cdb * fxf - sdb * fyf + cdrb * fxr - sdrb * fyr - rolling) / m
    beta_dot = -r + (sdb * fxf + cdb * fyf + sdrb * fxr + cdrb * fyr) / (m * v)
    w_dot = p.R * (tau - p.R * (fxf + fxr)) / max(p.J, 1e-6)
    body_ax = (cd * fxf - sd * fyf + cdr * fxr - sdr * fyr - cb * rolling) / m
    body_ay = (sd * fxf + cd * fyf + sdr * fxr + cdr * fyr - sb * rolling) / m
    return np.stack([yaw_acc, v_dot, beta_dot, w_dot], 1), body_ax, body_ay, (fzf, fzr)


def self_consistent_ax(x0, u0, p, iterations=8):
    ax_lt = np.zeros(x0.shape[0])
    for _ in range(iterations):
        _, ax_lt, _, _ = derivative(x0, u0, p, ax_lt)
    return ax_lt


def simulate(x0, controls, p, ax_lt0, dt=0.06, steps=6, substeps=70, ctrl_dt=0.01):
    """Euler rollout carrying ax_lt between substeps (lagged by p.tau_lt when positive).

    controls (N, M, 3) are the actual controls sampled every ctrl_dt from t0.
    Returns states (N, steps+1, 4) and body ax, ay and front load at each step.
    """
    n = x0.shape[0]
    h = dt / substeps
    x, ax_lt = x0.copy(), ax_lt0.copy()
    states = np.empty((n, steps + 1, 4))
    out_ax, out_ay, out_fzf = (np.empty((n, steps + 1)) for _ in range(3))
    states[:, 0] = x

    def control_at(t):
        return controls[:, min(int((t + 1e-9) / ctrl_dt), controls.shape[1] - 1)]

    _, out_ax[:, 0], out_ay[:, 0], (out_fzf[:, 0], _) = derivative(x, control_at(0.0), p, ax_lt)
    t = 0.0
    for k in range(steps):
        for _ in range(substeps):
            dx, body_ax, _, _ = derivative(x, control_at(t), p, ax_lt)
            if p.h:
                ax_lt = ax_lt + h / p.tau_lt * (body_ax - ax_lt) if p.tau_lt else body_ax
            x = x + h * dx
            t += h
        states[:, k + 1] = x
        _, out_ax[:, k + 1], out_ay[:, k + 1], (out_fzf[:, k + 1], _) = derivative(x, control_at(t), p, ax_lt)
    return states, out_ax, out_ay, out_fzf


def simulate_fixed_point(x0, controls, p, iterations=1, dt=0.06, steps=6, substeps=70, ctrl_dt=0.01):
    """Euler rollout with no load-transfer state: each derivative evaluates the tires at static
    loads, then `iterations` more times at the loads implied by the previous pass's body ax."""
    n = x0.shape[0]
    h = dt / substeps
    x = x0.copy()
    states = np.empty((n, steps + 1, 4))
    states[:, 0] = x
    t = 0.0
    for k in range(steps):
        for _ in range(substeps):
            u = controls[:, min(int((t + 1e-9) / ctrl_dt), controls.shape[1] - 1)]
            ax_lt = np.zeros(n)
            for _ in range(iterations + 1):
                dx, ax_lt, _, _ = derivative(x, u, p, ax_lt)
            x = x + h * dx
            t += h
        states[:, k + 1] = x
    return states
