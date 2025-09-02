from __future__ import annotations
from abc import abstractmethod
import numpy as np
from ..stats.base import StatsTracker
from .bandwidth import Bandwidth
from .grid import Grid
from .density import Density

class Kernel:
    """
    Base class for kernel density estimation. Tracks kernel parameters, grid, bandwidth, density
    estimates, and number of points observed.
    """
    def __init__(self, name, bw: Bandwidth=None, grid: Grid=None):
        self.name = name        # Name of the kernel
        self._bw = bw           # Bandwidth object for bandwidth calculation
        self._grid = grid       # Grid object for evaluation points
        self._mu = 0            # Placeholder for mean
        self.density = 0        # Stores current estimated density
        self.support = None     # Support points where density is evaluated
        self.n_points = 0       # Number of points observed

    @property
    def bw(self):
        """Return the current bandwidth value from the Bandwidth object."""
        return self._bw.get()

    @property
    def grid(self):
        """Return the Grid object associated with this kernel."""
        return self._grid

    @abstractmethod
    def _pdf(self, x, mu):
        """
        Abstract method for computing the kernel density for a given observation.
         => implemented for each specific kernel type (e.g., Gaussian, Cosine).
        """
        pass

    def estimate(self, x=None) -> Density:
        """
        Return the current kernel density estimate as a Density object.
        This method takes the state of the kernel (self.density, self.support, self.n_points)
        and wraps it into a Density object to provide a snapshot of the current kernel state.
        """
        return Density(self.density , self.support, self.n_points)

    def update(self, subject: StatsTracker):
        """Update the kernel estimate using a new observation from a StatsTracker."""
        # Increment the number of points observed.
        self.n_points += 1

        # Generate support points from the associated grid.
        self.support = self._grid.get(100)

        # Compute density contribution at support points using _pdf.
        dens = self._pdf(self.support, subject._state.astype(np.float64))
        
        # Ensure density array matches support shape in multivariate case
        if not isinstance(self.density, int) and self.support.shape[1] != self.density.shape[1]:
            for _ in range(self.support.shape[1] - self.density.shape[1]):
                self.density = np.c_[self.density, np.zeros((len(self.support),1))]
        
        # Update running density estimate with new contribution (online algorithm)
        self.density += dens / self.bw

    @abstractmethod
    def reset_kernel(self):
        """Reset the kernel state to initial values."""
        self.density = 0
        self.n_points = 0
        self.support = None
