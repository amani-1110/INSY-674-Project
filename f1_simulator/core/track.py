import math
import numpy as np

class Track:
    def __init__(self, name='demo'):
        self.name = name
        self.a, self.b = 155., 105.
        self.t = np.linspace(0, 2 * math.pi, 2401)
        self.points = np.column_stack((self.a * np.cos(self.t), self.b * np.sin(self.t)))
        self.seg = np.diff(self.points, axis=0)
        self.s = np.r_[0, np.cumsum(np.linalg.norm(self.seg, axis=1))]
        self.length = float(self.s[-1])
    def angle(self, s):
        return float(np.interp(s % self.length, self.s, self.t))
    def position(self, s):
        t = self.angle(s)
        return (self.a * math.cos(t), self.b * math.sin(t))
    def heading(self, s):
        t = self.angle(s)
        return math.atan2(self.b * math.cos(t), -self.a * math.sin(t))
    def curvature(self, s):
        t = self.angle(s)
        return self.a * self.b / ((self.a * math.sin(t))**2 + (self.b * math.cos(t))**2)**1.5
    def left_width(self, s): return 10.
    def right_width(self, s): return 10.
    def boundaries(self, s):
        x, y = self.position(s)
        h = self.heading(s)
        return ((x - 10 * math.sin(h), y + 10 * math.cos(h)), (x + 10 * math.sin(h), y - 10 * math.cos(h)))
    def project(self, x, y):
        p = np.array([x, y])
        u = np.clip(np.sum((p - self.points[:-1]) * self.seg, axis=1) / np.sum(self.seg**2, axis=1), 0, 1)
        q = self.points[:-1] + u[:, None] * self.seg
        i = int(np.argmin(np.sum((q - p)**2, axis=1)))
        tangent = self.seg[i] / np.linalg.norm(self.seg[i])
        displacement = p - q[i]
        # Signed 2D cross product: positive on the left of the track tangent.
        offset = tangent[0] * displacement[1] - tangent[1] * displacement[0]
        return (float(self.s[i] + u[i] * (self.s[i+1] - self.s[i])), float(offset))
