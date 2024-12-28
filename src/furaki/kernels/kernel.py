from __future__ import annotations
from abc import abstractmethod
import numpy as np
from ..stats.base import StatsTracker
from .bandwidth import Bandwidth
from .grid import Grid
from .density import Density

class Kernel:
    def __init__(self, name, bw: Bandwidth=None, grid: Grid=None):
        self.name = name
        self._bw = bw
        self._grid = grid
        self._mu = 0
        self.density = 0
        self.support = None
        self.n_points = 0

    @property
    def bw(self):
        return self._bw.get()

    @property
    def grid(self):
        return self._grid

    @abstractmethod
    def _pdf(self, x, mu):
        pass

    def estimate(self, x=None) -> Density:
        return Density(self.density , self.support, self.n_points)

    def update(self, subject: StatsTracker):
        self.n_points += 1
        self.support = self._grid.get(100)
        dens = self._pdf(self.support, subject._state.astype(np.float64))
        
        if not isinstance(self.density, int) and self.support.shape[1] != self.density.shape[1]:
            for _ in range(self.support.shape[1] - self.density.shape[1]):
                self.density = np.c_[self.density, np.zeros((len(self.support),1))]
            
        self.density += dens / self.bw

    @abstractmethod
    def reset_kernel(self):
        self.density = 0
        self.n_points = 0
        self.support = None
