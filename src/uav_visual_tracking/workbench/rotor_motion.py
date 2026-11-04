"""Exact integration of a continuous piecewise-linear omega, in degrees."""
import numpy as np


class RotorMotion:
    def __init__(self, cfg):
        if cfg['mode'] != 'piecewise_linear_omega':
            raise ValueError('Unsupported rotor mode')
        self.points = np.asarray(cfg['omega_points_deg_s'], float)
        if (self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2
                or not np.isfinite(self.points).all() or self.points[0, 0] != 0
                or np.any(np.diff(self.points[:, 0]) <= 0)):
            raise ValueError('omega points must be finite, start at 0 and strictly increase in time')
        self.theta0 = float(cfg['theta0_deg'])
        if not np.isfinite(self.theta0):
            raise ValueError('theta0 must be finite')

    def sample(self, t):
        t = np.asarray(t, float)
        if np.any(t < 0):
            raise ValueError('Negative rotor time')
        phase = np.full_like(t, self.theta0)
        for (a, wa), (b, wb) in zip(self.points[:-1], self.points[1:]):
            d = np.clip(t-a, 0, b-a)
            phase += wa*d + .5*(wb-wa)/(b-a)*d*d
        phase += self.points[-1, 1]*np.maximum(t-self.points[-1, 0], 0)
        return phase, np.interp(t, self.points[:, 0], self.points[:, 1])
