import numpy as np
from ..kernels import Kernel, Bandwidth, Grid

class Gaussian(Kernel):
    def __init__(self, bw: Bandwidth=None, grid: Grid=None):
        super().__init__("gaussian", bw, grid)

    def _pdf(self, x, mu):
        u = (x - mu) / self.bw
        return 1/np.sqrt(2*np.pi) * np.exp(-0.5 * (u * u))
