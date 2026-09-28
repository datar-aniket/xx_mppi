"""Shared data handling for the load-transfer analysis: alignment, selection, CG acceleration."""

import glob
import os

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, savgol_filter

G = 9.80665
FS = 100.0  # /ekf/state rate
EKF_COLUMNS = ['t_bag', 'stamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
               'wx', 'wy', 'wz', 'beta', 'current', 'steer', 'erpm', 'steer_rear', 'solution', 'reset',
               'source', 'armed', 'auto', 'trigger', 'tau', 'vw']
E = {name: i for i, name in enumerate(EKF_COLUMNS)}

# IMU position relative to base_link (vehicle centre, taken as the CG). The IMU
# lever arm in calibrations/config/carxx/calibration/imu_calibration_params.yaml
# is (0.0696, 0.0083, -0.0213) m from the carxx frame (rear axle), and base_link
# is 0.13 m ahead of the rear axle. The MCU publishes IMU acceleration without
# lever-arm compensation, so r^2 * 0.06 m (0.5 m/s^2 at 3 rad/s) reads as forward
# acceleration in corners unless removed.
P_IMU = np.array([0.0696 - 0.13, 0.0083, -0.0213])


def lowpass(x, fc, fs=FS, order=2):
    """Zero-phase Butterworth low pass."""
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x)


def lowpass_causal(x, fc, fs=FS, order=2):
    b, a = butter(order, fc / (fs / 2))
    return lfilter(b, a, x)


def quaternion_euler(qx, qy, qz, qw):
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    # Third row of the body -> ENU rotation: gravity direction in the body frame.
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx**2 + qy**2)
    return roll, pitch, yaw, (r20, r21, r22)


def align(npz_path):
    """One bag on the EKF header time base. Pitch is FLU (+ = nose down)."""
    data = np.load(npz_path)
    e, dc = data['ekf'], data['dc']
    t = e[:, E['stamp']]
    keep = np.concatenate([[True], np.diff(t) > 1e-4])
    e, t = e[keep], t[keep]
    roll, pitch, yaw, (r20, r21, r22) = quaternion_euler(e[:, E['qx']], e[:, E['qy']], e[:, E['qz']], e[:, E['qw']])
    o = dict(t=t, auto=e[:, E['auto']] > 0.5, vx=e[:, E['vx']], vy=e[:, E['vy']],
             v=np.hypot(e[:, E['vx']], e[:, E['vy']]), vw=e[:, E['vw']],
             ax=e[:, E['ax']], ay=e[:, E['ay']], roll=roll, pitch=pitch, yaw=np.unwrap(yaw),
             wx=e[:, E['wx']], wy=e[:, E['wy']], wz=e[:, E['wz']], tau=e[:, E['tau']],
             steer_fb=e[:, E['steer']], beta=e[:, E['beta']], reset=e[:, E['reset']])
    # The published acceleration is (accelerometer - bias) - R^T g, so the
    # bias-corrected specific force is a + g * (third row of R).
    o['fx'] = o['ax'] + G * r20
    # Commands, zero-order hold. Command bag times move to the EKF header clock
    # with the median bag-minus-header offset of the EKF messages.
    offset = np.median(e[:, E['t_bag']] - t)
    for key in ('tau_cmd', 'steer_cmd', 'rsteer_cmd'):
        o[key] = np.full(len(t), np.nan)
    o['cmd_age'] = np.full(len(t), np.inf)
    if len(dc):
        t_cmd = dc[:, 0] - offset
        idx = np.searchsorted(t_cmd, t, side='right') - 1
        ok = idx >= 0
        o['tau_cmd'][ok] = dc[idx[ok], 4]
        o['steer_cmd'][ok] = dc[idx[ok], 2]
        o['rsteer_cmd'][ok] = dc[idx[ok], 3]
        o['cmd_age'][ok] = t[ok] - t_cmd[idx[ok]]
    o['a_wheel'] = savgol_filter(o['vw'], 15, 3, deriv=1, delta=1.0 / FS)
    rdot = np.gradient(lowpass(o['wz'], 10), 1.0 / FS)
    r = o['wz']
    o['ax_cg'] = o['ax'] + rdot * P_IMU[1] + r**2 * P_IMU[0]
    o['ay_cg'] = o['ay'] - rdot * P_IMU[0] + r**2 * P_IMU[1]
    return o


def load_all(data_dir):
    bags = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    return {os.path.splitext(os.path.basename(p))[0]: align(p) for p in bags}


def auto_segments(auto):
    segments, start = [], None
    for i, a in enumerate(auto):
        if a and start is None:
            start = i
        if not a and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(auto) - 1))
    return segments


def valid_mask(o, vmin=2.0, crash_mps2=12.0, pre_crash_s=2.0, head_s=0.5, tail_s=1.5):
    """AUTO driving above vmin, without crash windows.

    A crash is |(ax, ay)| above crash_mps2 after a 10 Hz low pass, more than
    twice what the tires can produce. Everything from pre_crash_s before the
    first one to the end of that AUTO segment is dropped, as are the first
    head_s and last tail_s of every segment (operator hand-over).
    """
    t = o['t']
    horizontal = np.hypot(lowpass(o['ax'], 10), lowpass(o['ay'], 10))
    m = np.zeros(len(t), bool)
    for s, e in auto_segments(o['auto']):
        seg = np.zeros(len(t), bool)
        seg[s:e + 1] = True
        seg &= (t >= t[s] + head_s) & (t <= t[e] - tail_s)
        crash = np.flatnonzero(horizontal[s:e + 1] > crash_mps2)
        if len(crash):
            seg &= t < t[s + crash[0]] - pre_crash_s
        m |= seg
    m &= (o['v'] > vmin) & (o['vw'] > vmin)
    m &= np.concatenate([[True], np.diff(o['reset']) == 0])
    for key in ('ax', 'ay', 'fx', 'pitch', 'roll', 'wy', 'wx', 'wz', 'tau_cmd', 'steer_cmd', 'rsteer_cmd',
                'a_wheel', 'vw', 'v', 'tau'):
        m &= np.isfinite(o[key])
    m &= o['cmd_age'] < 0.1
    return m


def contiguous(m, min_len):
    runs, start = [], None
    for i, x in enumerate(m):
        if x and start is None:
            start = i
        if not x and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(m) - start >= min_len:
        runs.append((start, len(m)))
    return runs


def servo(cmd, dead_samples=6, alpha=0.5):
    """Steering servo: dead time then a first-order lag (calibrated ~60 ms, ~10 ms)."""
    out = np.empty_like(cmd)
    shifted = np.roll(cmd, dead_samples)
    shifted[:dead_samples] = np.nan
    finite = np.flatnonzero(np.isfinite(cmd))
    y = cmd[finite[0]] if len(finite) else 0.0
    for i, v in enumerate(shifted):
        if np.isfinite(v):
            y = y + alpha * (v - y)
        out[i] = y
    return out
