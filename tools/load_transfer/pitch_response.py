#!/usr/bin/env python3
"""How much does the chassis pitch with longitudinal acceleration, and is that pitch usable as a
load-transfer measurement?

Works on the .npz files from extract_bags.py, using AUTO driving above 2 m/s with
crash windows removed (lt_data.valid_mask). Prints the low-frequency pitch gradient
(pitch against CG acceleration, both low-passed at 2 Hz and de-meaned per run),
its bag-bootstrap interval, a lag scan, the same fit on gyro-integrated pitch
(to separate chassis rotation from EKF tilt corrections), and roll against
lateral acceleration for comparison.

  ./pitch_response.py --data /tmp/load_transfer [--plots /tmp/load_transfer/plots]
"""

import argparse
import os
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid

import lt_data

LOWPASS_HZ = 2.0
MIN_RUN = 300  # samples


def runs_table(bags):
    """Low-passed, per-run signals for the regressions."""
    rows = []
    for name, o in bags.items():
        m = lt_data.valid_mask(o)
        lp = {k: lt_data.lowpass(np.nan_to_num(o[k]), LOWPASS_HZ) for k in ('ax_cg', 'ay_cg', 'pitch', 'roll')}
        # Euler pitch rate from the gyro; wy alone picks up yaw rate through roll.
        pitch_rate = o['wy'] * np.cos(o['roll']) - o['wz'] * np.sin(o['roll'])
        for s, e in lt_data.contiguous(m, MIN_RUN):
            t = o['t'][s:e] - o['t'][s]
            gyro_pitch = cumulative_trapezoid(pitch_rate[s:e], t, initial=0.0)
            correction = (o['pitch'][s:e] - o['pitch'][s]) - gyro_pitch
            # Linear detrend removes the gyro bias ramp.
            gyro_pitch -= np.polyval(np.polyfit(t, gyro_pitch, 1), t)
            correction -= np.polyval(np.polyfit(t, correction, 1), t)
            rows.append(dict(bag=name, ax=lp['ax_cg'][s:e], ay=lp['ay_cg'][s:e], pitch=lp['pitch'][s:e],
                             roll=lp['roll'][s:e], gyro_pitch=lt_data.lowpass(gyro_pitch, LOWPASS_HZ),
                             correction=lt_data.lowpass(correction, LOWPASS_HZ)))
    return rows


def gain(rows, y_key, x_key, lag=0):
    """Least-squares slope of y on x (per-run de-meaned); positive lag delays y relative to x."""
    xs, ys = [], []
    for r in rows:
        x, y = r[x_key], r[y_key]
        if lag > 0:
            x, y = x[:-lag], y[lag:]
        elif lag < 0:
            x, y = x[-lag:], y[:lag]
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    k = (x @ y) / (x @ x)
    residual = y - k * x
    return k, 1.0 - residual.var() / y.var(), y.std(), residual.std(), x.std()


def step_averages(bags, change_nm=0.2, pre=40, post=80):
    """Average ax and pitch around torque-command steps (100 ms change beyond change_nm)."""
    events = {'brake': [], 'throttle': []}
    for o in bags.values():
        m = lt_data.valid_mask(o)
        tau = o['tau_cmd']
        d = np.full(len(tau), np.nan)
        d[10:] = tau[10:] - tau[:-10]
        last = -1000
        for i in range(pre + 10, len(tau) - post):
            if i - last < 50 or not m[i - 20:i + 60].all():
                continue
            for kind, hit in (('brake', d[i] < -change_nm), ('throttle', d[i] > change_nm)):
                if hit and abs(d[i]) >= np.nanmax(np.abs(d[i - 5:i + 6])) - 1e-9:
                    w = slice(i - pre, i + post)
                    base = slice(i - 25, i - 10)
                    events[kind].append(dict(tau=tau[w] - tau[i - 10], ax=o['ax_cg'][w] - o['ax_cg'][base].mean(),
                                             pitch=o['pitch'][w] - o['pitch'][base].mean()))
                    last = i
    t = np.arange(-pre, post) / lt_data.FS
    return t, {k: {s: np.array([e[s] for e in v]) for s in ('tau', 'ax', 'pitch')} for k, v in events.items() if v}


def plots(bags, rows, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import csd, welch, detrend
    blue, orange = '#2a78d6', '#eb6834'
    plt.rcParams.update({'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.6, 'font.size': 9,
                         'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#898781',
                         'ytick.color': '#898781', 'legend.frameon': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    os.makedirs(out_dir, exist_ok=True)
    # Transfer function CG ax -> EKF pitch and -> gyro-integrated pitch (Welch, 2.56 s segments).
    nps = 256
    acc = {}
    for o in bags.values():
        m = lt_data.valid_mask(o)
        for s, e in lt_data.contiguous(m, nps + 64):
            x = detrend(o['ax_cg'][s:e])
            for key, y in (('pitch', detrend(o['pitch'][s:e])), ('wy', detrend(o['wy'][s:e]))):
                f, pxy = csd(x, y, lt_data.FS, nperseg=nps)
                _, pxx = welch(x, lt_data.FS, nperseg=nps)
                _, pyy = welch(y, lt_data.FS, nperseg=nps)
                for part, val in (('xy', pxy), ('xx', pxx), ('yy', pyy)):
                    acc.setdefault((key, part), 0)
                    acc[(key, part)] = acc[(key, part)] + val * (e - s)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for key, color, label in (('pitch', blue, 'EKF pitch'), ('wy', orange, '∫ gyro pitch rate')):
        h = acc[(key, 'xy')] / acc[(key, 'xx')]
        if key == 'wy':
            h = h / (2j * np.pi * np.maximum(f, 1e-9))
        coherence = np.abs(acc[(key, 'xy')])**2 / (acc[(key, 'xx')] * acc[(key, 'yy')])
        ax[0].semilogx(f[1:], np.degrees(np.abs(h[1:])), color=color, label=label)
        ax[1].semilogx(f[1:], coherence[1:], color=color, label=label)
    ax[0].set_ylabel('|pitch / ax| (° per m/s²)')
    ax[1].set_ylabel('coherence')
    ax[1].set_xlabel('frequency (Hz)')
    ax[1].set_ylim(0, 1)
    ax[0].legend()
    ax[0].set_title('Pitch response to CG longitudinal acceleration')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pitch_transfer_function.png'), dpi=100)
    # Step-triggered averages.
    t, events = step_averages(bags)
    fig, ax = plt.subplots(3, len(events), figsize=(6 * len(events), 7), sharex=True, squeeze=False)
    for col, (kind, ev) in enumerate(sorted(events.items())):
        n = len(ev['ax'])
        for row, (key, scale, label) in enumerate((('tau', 1.0, 'Δ torque command (N m)'),
                                                   ('ax', 1.0, 'Δ CG ax (m/s²)'),
                                                   ('pitch', np.degrees(1.0), 'Δ pitch (°, + = nose down)'))):
            mean = ev[key].mean(0) * scale
            sem = ev[key].std(0) / np.sqrt(n) * scale
            ax[row, col].fill_between(t, mean - 2 * sem, mean + 2 * sem, color=blue, alpha=0.15, lw=0)
            ax[row, col].plot(t, mean, color=blue, lw=1.5)
            ax[row, col].set_ylabel(label)
        ax[0, col].set_title(f'{kind} steps (n = {n})')
        ax[2, col].set_xlabel('time from step (s)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pitch_step_averages.png'), dpi=100)
    print(f'plots written to {out_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='directory with the extract_bags.py .npz files')
    parser.add_argument('--plots', help='directory for transfer-function and step-average plots')
    args = parser.parse_args()
    bags = lt_data.load_all(args.data)
    total = 0.0
    for name, o in bags.items():
        seconds = lt_data.valid_mask(o).sum() / lt_data.FS
        total += seconds
        print(f'{name:26s} {seconds:6.1f} s valid')
    print(f'{"total":26s} {total:6.1f} s')
    rows = runs_table(bags)
    print(f'\n{len(rows)} runs of at least {MIN_RUN / lt_data.FS:.0f} s, signals low-passed at {LOWPASS_HZ} Hz')

    k, r2, sy, sr, sx = gain(rows, 'pitch', 'ax')
    print(f'EKF pitch ~ CG ax: {np.degrees(k):+.4f} °/(m/s²)  R² {r2:.3f}  '
          f'pitch std {np.degrees(sy):.3f}°  residual {np.degrees(sr):.3f}°')
    bag_names = sorted({r['bag'] for r in rows})
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(400):
        pick = rng.choice(bag_names, len(bag_names))
        boot.append(np.degrees(gain([r for b in pick for r in rows if r['bag'] == b], 'pitch', 'ax')[0]))
    print(f'  bag-bootstrap 95% interval [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] °/(m/s²)')
    print(f'  as an ax sensor: {np.degrees(sr) / abs(np.degrees(k)):.1f} m/s² of noise (1σ) '
          f'against {sx:.2f} m/s² of signal')
    print('  lag scan (pitch delayed by):', '  '.join(
        f'{lag * 10:+d} ms {np.degrees(gain(rows, "pitch", "ax", lag)[0]):+.4f} (R² {gain(rows, "pitch", "ax", lag)[1]:.3f})'
        for lag in (-10, 0, 5, 10, 20, 40)))
    k, r2, *_ = gain(rows, 'gyro_pitch', 'ax')
    print(f'gyro-integrated pitch ~ CG ax: {np.degrees(k):+.4f} °/(m/s²)  R² {r2:.3f}')
    k, r2, *_ = gain(rows, 'correction', 'ax')
    print(f'EKF minus gyro pitch (tilt corrections) ~ CG ax: {np.degrees(k):+.4f} °/(m/s²)  R² {r2:.3f}')
    k, r2, sy, sr, _ = gain(rows, 'roll', 'ay')
    print(f'roll ~ CG ay: {np.degrees(k):+.4f} °/(m/s²)  R² {r2:.3f}')
    print('per bag, pitch ~ CG ax:')
    for b in bag_names:
        sub = [r for r in rows if r['bag'] == b]
        k, r2, *_ = gain(sub, 'pitch', 'ax')
        print(f'  {b:26s} {np.degrees(k):+.4f} °/(m/s²)  R² {r2:.3f}  n {sum(len(r["ax"]) for r in sub)}')
    t, events = step_averages(bags)
    for kind, ev in sorted(events.items()):
        w = (t > 0.0) & (t < 0.3)
        mean_ax, mean_pitch = ev['ax'].mean(0), np.degrees(ev['pitch'].mean(0))
        i_ax = np.argmax(np.abs(mean_ax[w]))
        i_p = np.argmax(np.abs(mean_pitch[w]))
        print(f'{kind} steps (n={len(ev["ax"])}): peak Δax {mean_ax[w][i_ax]:+.2f} m/s² at {t[w][i_ax] * 1e3:.0f} ms, '
              f'peak Δpitch {mean_pitch[w][i_p]:+.3f}° at {t[w][i_p] * 1e3:.0f} ms')
    if args.plots:
        plots(bags, rows, args.plots)
    return 0


if __name__ == '__main__':
    sys.exit(main())
