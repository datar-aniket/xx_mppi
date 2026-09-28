#!/usr/bin/env python3
"""Replay the MPPI Fiala body model on logged AUTO driving and fit the load-transfer height.

From every 0.1 s of valid driving (lt_data.valid_mask) the model starts at the
measured state (EKF yaw rate, speed and sideslip, VESC wheel speed) and runs
open loop for 6 x 0.06 s with the logged commands: torque as sent, steering
through the calibrated servo dead time unless --no-servo-delay. Predictions are
scored against the measured yaw rate, speed and IMU acceleration at the CG.

Fits the tire balance (Cf, Cr, mu_f, mu_r) with h pinned at zero, then with h
free, on --fit-bags, and reports the other bags as held out. Then profiles h,
shows the yaw error against acceleration with and without load transfer,
compares carried and lagged load-transfer states with the ways of initializing
them (model, IMU ax, pitch, zero), and checks that one fixed-point pass inside
the derivative matches the carried state.

  ./fit_load_transfer.py --data /tmp/load_transfer --figure summary.png
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

import lt_data
from fiala_model import (CARXX_IROS, Params, params_from_vehicle_yaml, self_consistent_ax, simulate,
                         simulate_fixed_point)

DT, STEPS, SUBSTEPS = 0.06, 6, 70  # carxx mppi.yaml dt and integration_substeps
HORIZON = int(round(DT * STEPS * lt_data.FS)) + 1
STRIDE = 10  # samples between replay starts
W_AY = 0.1  # weight of the lateral acceleration error in the fit loss
PITCH_GRADIENT = np.radians(-0.0396)  # rad per m/s^2, from pitch_response.py
DEFAULT_FIT_BAGS = 'iros01,iros02,iros03_horizon01,iros04,iros05,iros06'


def build_dataset(bags, servo_delay=True):
    ds = {k: [] for k in ('x0', 'ctrl', 'meas', 'ax_imu0', 'pitch_dev0', 'bag')}
    step = int(round(DT * lt_data.FS))
    for name, o in bags.items():
        m = lt_data.valid_mask(o)
        if not m.any():
            continue
        ax_lp, ay_lp = lt_data.lowpass(o['ax_cg'], 5), lt_data.lowpass(o['ay_cg'], 5)
        ax_causal = lt_data.lowpass_causal(np.nan_to_num(o['ax_cg']), 5)
        pitch_causal = lt_data.lowpass_causal(o['pitch'] - np.median(o['pitch'][m]), 5)
        dead, alpha = (6, 0.5) if servo_delay else (0, 1.0)
        front, rear = lt_data.servo(o['steer_cmd'], dead, alpha), lt_data.servo(o['rsteer_cmd'], dead, alpha)
        beta = np.clip(np.nan_to_num(o['beta']), -2.5, 2.5)
        for i in range(0, len(o['t']) - HORIZON, STRIDE):
            if not m[i:i + HORIZON].all():
                continue
            ks = i + np.arange(STEPS + 1) * step
            ds['x0'].append([o['wz'][i], o['v'][i], beta[i], o['vw'][i]])
            ds['ctrl'].append(np.stack([front[i:i + HORIZON], o['tau_cmd'][i:i + HORIZON], rear[i:i + HORIZON]], 1))
            ds['meas'].append(np.stack([o['wz'][ks], o['v'][ks], beta[ks], o['vw'][ks], ax_lp[ks], ay_lp[ks]], 1))
            ds['ax_imu0'].append(ax_causal[i])
            ds['pitch_dev0'].append(pitch_causal[i])
            ds['bag'].append(name)
    return {k: np.array(v) for k, v in ds.items()}


def run(ds, p, init='model', sel=slice(None)):
    x0, ctrl, meas = ds['x0'][sel], ds['ctrl'][sel], ds['meas'][sel]
    if init == 'zero':
        lt0 = np.zeros(len(x0))
    elif init == 'imu':
        lt0 = ds['ax_imu0'][sel]
    elif init == 'pitch':
        lt0 = ds['pitch_dev0'][sel] / PITCH_GRADIENT
    else:
        lt0 = self_consistent_ax(x0, ctrl[:, 0], p)
    states, ax, ay, fzf = simulate(x0, ctrl, p, lt0, DT, STEPS, SUBSTEPS)
    err = {key: states[:, :, i] - meas[:, :, i] for i, key in enumerate(('r', 'V', 'beta', 'w'))}
    err['ax'], err['ay'] = ax - meas[:, :, 4], ay - meas[:, :, 5]
    return err, fzf


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


def params_from(z, base, fix_h=None):
    h, log_cf, log_cr, log_muf, log_mur = z
    return Params(base, h=h if fix_h is None else fix_h, Cf=np.exp(log_cf), Cr=np.exp(log_cr),
                  muf=np.exp(log_muf), mur=np.exp(log_mur))


def loss(z, ds, sel, base, fix_h=None):
    p = params_from(z, base, fix_h)
    if not -0.05 <= p.h <= 0.25:
        return 1e3
    err, _ = run(ds, p, 'model', sel)
    return float(np.mean(err['r'][:, 1:]**2) + W_AY * np.mean(err['ay'][:, 1:]**2))


def score(ds, p, split):
    parts = []
    for name, sel in split:
        err, _ = run(ds, p, 'model', sel)
        parts.append(f'{name}: yaw {rms(err["r"][:, 1:]):.3f} (@0.36 s {rms(err["r"][:, 6]):.3f}) '
                     f'ay {rms(err["ay"][:, 1:]):.3f} V {rms(err["V"][:, 1:]):.3f}')
    return ' | '.join(parts)


def describe(p):
    return f'h {p.h:.3f} m  Cf {p.Cf:.0f}  Cr {p.Cr:.0f}  mu_f {p.muf:.3f}  mu_r {p.mur:.3f}'


def ax_binned_yaw_error(ds, p, bins=(-4.5, -3, -1.5, -0.5, 0.5, 1.5, 3, 4.5), min_ay=2.0):
    """Mean yaw-rate error at the horizon end, signed so + is over-rotation, by mean measured ax."""
    meas = ds['meas']
    ax_mean = meas[:, 1:, 4].mean(1)
    turning = np.sign(meas[:, 0, 0] + 1e-9)
    cornering = np.abs(meas[:, 0, 5]) > min_ay
    err, _ = run(ds, p, 'model')
    e = err['r'][:, -1] * turning
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = cornering & (ax_mean >= lo) & (ax_mean < hi)
        out.append((0.5 * (lo + hi), e[s].mean(), e[s].std() / np.sqrt(max(s.sum(), 1)), int(s.sum())))
    slope = np.polyfit(ax_mean[cornering], e[cornering], 1)[0]
    return out, slope


def summary_figure(path, bags, ds, p_fit, h_best, profile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pitch_response
    blue, orange, ink = '#2a78d6', '#eb6834', '#52514e'
    plt.rcParams.update({'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.6, 'font.size': 9,
                         'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': ink, 'xtick.color': '#898781',
                         'ytick.color': '#898781', 'legend.frameon': False, 'lines.linewidth': 2.0,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = pitch_response.runs_table(bags)
    k = pitch_response.gain(rows, 'pitch', 'ax')[0]
    x = np.concatenate([r['ax'] - r['ax'].mean() for r in rows])
    y = np.degrees(np.concatenate([r['pitch'] - r['pitch'].mean() for r in rows]))
    edges = np.arange(-6, 6.1, 1.0)
    centres, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() > 200:
            centres.append(0.5 * (lo + hi))
            means.append(y[s].mean())
            stds.append(y[s].std())
    centres, means, stds = map(np.array, (centres, means, stds))
    ax[0].fill_between(centres, means - stds, means + stds, color=blue, alpha=0.15, lw=0, label='±1σ in bin')
    ax[0].plot(centres, means, 'o-', color=blue, ms=5, label='mean in bin')
    ax[0].plot([-6, 6], [np.degrees(k) * -6, np.degrees(k) * 6], '--', color=ink, lw=1.2,
               label=f'fit {np.degrees(k):+.3f} °/(m/s²)')
    ax[0].set_xlabel('CG longitudinal acceleration (m/s², 2 Hz low pass)')
    ax[0].set_ylabel('pitch change (°, + = nose down)')
    ax[0].set_title('Pitch barely moves with acceleration')
    ax[0].legend(loc='upper right')
    for h, color, label in ((0.0, orange, 'no load transfer'), (h_best, blue, f'load transfer, h = {h_best:.3f} m')):
        bins, _ = ax_binned_yaw_error(ds, Params(p_fit, h=h))
        c, mean, sem, _ = map(np.array, zip(*bins))
        ax[1].errorbar(c, mean, yerr=2 * sem, fmt='o-', color=color, ms=5, elinewidth=1, label=label)
    ax[1].axhline(0, color='#c3c2b7', lw=1)
    ax[1].set_xlabel('mean measured ax over the 0.36 s horizon (m/s²)')
    ax[1].set_ylabel('yaw-rate error at 0.36 s (rad/s, + = over-rotates)')
    ax[1].set_title('Missing load transfer shows up in yaw prediction')
    ax[1].legend(loc='upper left')
    hs, fit_rms, test_rms = map(np.array, zip(*profile))
    ax[2].plot(hs, fit_rms, 'o-', color=blue, ms=5, label='fit bags')
    ax[2].plot(hs, test_rms, 'o-', color=orange, ms=5, label='held-out bags')
    ax[2].set_xlabel('load-transfer height h (m)')
    ax[2].set_ylabel('yaw-rate RMSE over 0.06–0.36 s (rad/s)')
    ax[2].set_title('Yaw prediction error against h')
    ax[2].legend(loc='upper center')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(f'figure written to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='directory with the extract_bags.py .npz files')
    parser.add_argument('--vehicle', help='vehicle.yaml for the starting parameters (default: carxx at IROS)')
    parser.add_argument('--fit-bags', default=DEFAULT_FIT_BAGS, help='comma-separated bags to fit on')
    parser.add_argument('--no-servo-delay', action='store_true', help='apply steering commands instantly, '
                        'as the MPPI model does, instead of through the servo dead time')
    parser.add_argument('--figure', help='write the summary figure here')
    parser.add_argument('--json', help='write the fitted parameters here')
    args = parser.parse_args()

    base = params_from_vehicle_yaml(args.vehicle) if args.vehicle else Params(CARXX_IROS)
    bags = lt_data.load_all(args.data)
    ds = build_dataset(bags, servo_delay=not args.no_servo_delay)
    fit = np.isin(ds['bag'], args.fit_bags.split(','))
    split = (('fit', fit), ('held-out', ~fit))
    print(f'{len(ds["x0"])} replay starts ({fit.sum()} fit, {(~fit).sum()} held out), '
          f'servo delay {"off" if args.no_servo_delay else "on"}')
    print('yaw rate in rad/s, ay in m/s^2, V in m/s; RMSE over steps 1-6 unless marked')
    print(f'starting parameters, {describe(Params(base, h=0.0))}\n  {score(ds, Params(base, h=0.0), split)}')

    t0 = time.time()
    z0 = np.array([0.0, np.log(base.Cf), np.log(base.Cr), np.log(base.muf), np.log(base.mur)])
    res = minimize(lambda z: loss(np.concatenate([[0.0], z]), ds, fit, base, 0.0), z0[1:], method='Nelder-Mead',
                   options=dict(maxiter=160, xatol=1e-3, fatol=1e-5))
    z_static = np.concatenate([[0.0], res.x])
    p_static = params_from(z_static, base, 0.0)
    print(f'tire balance fitted, no load transfer: {describe(p_static)}\n  {score(ds, p_static, split)}')
    z1 = z_static.copy()
    z1[0] = 0.05
    res = minimize(lambda z: loss(z, ds, fit, base), z1, method='Nelder-Mead',
                   options=dict(maxiter=260, xatol=1e-3, fatol=1e-5))
    p_fit = params_from(res.x, base)
    print(f'tire balance and h fitted: {describe(p_fit)}\n  {score(ds, p_fit, split)}  [{time.time() - t0:.0f} s]')

    profile = []
    print('h profile with those tires (yaw RMSE fit / held-out):')
    for h in (0.0, 0.02, 0.035, 0.045, 0.055, 0.065, 0.08, 0.10, 0.13):
        values = [rms(run(ds, Params(p_fit, h=h), 'model', sel)[0]['r'][:, 1:]) for _, sel in split]
        profile.append((h, *values))
        print(f'  h {h:.3f}: {values[0]:.4f} / {values[1]:.4f}')
    h_best = min(profile, key=lambda row: row[2])[0]
    print(f'  held-out optimum h = {h_best:.3f} m')

    for h in (0.0, p_fit.h):
        bins, slope = ax_binned_yaw_error(ds, Params(p_fit, h=h))
        print(f'yaw error at 0.36 s (+ = over-rotates) by mean ax, |ay| > 2, h = {h:.3f}: slope {slope:+.4f} '
              f'(rad/s)/(m/s²)\n  ' + '  '.join(f'{c:+.2f}: {m:+.3f}±{s:.3f} (n {n})' for c, m, s, n in bins))

    print(f'load-transfer lag and initialization at h = {p_fit.h:.3f} (yaw RMSE, all starts):')
    for tau in (0.0, 0.03, 0.06, 0.12, 0.25):
        inits = ('model',) if tau == 0.0 else ('model', 'imu', 'pitch', 'zero')
        cells = [f'{init} {rms(run(ds, Params(p_fit, tau_lt=tau), init)[0]["r"][:, 1:]):.4f}' for init in inits]
        print(f'  tau {tau:.2f} s: ' + '  '.join(cells))

    carried = rms(run(ds, p_fit, 'model')[0]['r'][:, 1:])
    for iterations in (0, 1, 2):
        states = simulate_fixed_point(ds['x0'], ds['ctrl'], p_fit, iterations, DT, STEPS, SUBSTEPS)
        value = rms(states[:, 1:, 0] - ds['meas'][:, 1:, 0])
        label = 'static loads' if iterations == 0 else f'{iterations} fixed-point pass(es)'
        print(f'  in-derivative, {label}: yaw RMSE {value:.4f} (carried state {carried:.4f})')

    if args.figure:
        summary_figure(args.figure, bags, ds, p_fit, round(float(p_fit.h), 3), profile)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(static=dict(p_static), fitted=dict(p_fit), held_out_best_h=h_best,
                           profile=profile, fit_bags=args.fit_bags.split(','),
                           servo_delay=not args.no_servo_delay), f, indent=2, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
