import numpy as np
from ..kernels import Kernel, Bandwidth, Grid

class Cosine(Kernel):
    def __init__(self, bw: Bandwidth=None, grid: Grid=None):
        super().__init__("cosine", bw, grid)

    def _pdf(self, x, mu):
        u = (x-mu)/self.bw
        dens = (np.pi/4 * np.cos(np.pi/2 * u))
        dens[abs(u) > 1] = 1e-12
        return dens