from ..kernels import Kernel, Bandwidth, Grid

class Epanechnikov(Kernel):
    """Epanechnikov kernel implementation."""
    def __init__(self, bw: Bandwidth=None, grid: Grid=None):
        super().__init__("epanechnikov", bw, grid)

    def _pdf(self, x, mu):
        u = (x-mu)/self.bw
        dens = (0.75 * (1 - (u * u)))
        dens[abs(u) > 1] = 0
        return dens
