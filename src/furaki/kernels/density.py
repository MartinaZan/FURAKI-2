import numpy as np
from .grid import Grid

class Density:
    def __init__(self, values, grid: Grid, n_points=100) -> None:
        self.y = np.log(values + 1e-12)
        self.x = grid
        self.is_normalized = False
        self.nc = 1. # normalization constant
        self.n_points = n_points

    @property
    def shape(self):
        return self.y.shape
    
    def __getitem__(self, idx):
        return self.y[idx]
    
    def get(self, normalize=True, as_prod=True):
        dens = self.y
        
        if normalize:
            if self.x is None:
                dens -= np.trapz(dens, axis=0)
            else:
                dens -= np.trapz(dens, self.x, axis=0)
        
        if as_prod:
            if dens.ndim > 1:
                dens = np.sum(dens, axis=1)
        
        return np.log(self.nc) + dens - np.log(self.n_points)
    
    def cdf(self, x):
        d = len(x)
        low = np.min(self.x,  axis=0)
        support = np.linspace(low, x, num=100)
        dens = np.exp(self.y)
        cumulative = np.array([np.trapz(dens, support[:, i]) for i in range(d)], dtype=np.float64)
        return cumulative
