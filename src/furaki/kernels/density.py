import numpy as np
from .grid import Grid

class Density:
    """Class to store and manipulate a kernel density estimate (KDE)."""
    def __init__(self, values, grid: Grid, n_points=100) -> None:
        self.y = np.log(values + 1e-12) # Logaritm of density values
        self.x = grid                   # Support points (grid) where density is evaluated
        self.is_normalized = False      # Flag indicating if density is normalized
        self.nc = 1.                    # Normalization constant (multiplicative)
        self.n_points = n_points        # Number of observations used

    @property
    def shape(self):
        """Return the shape of the log-density array."""
        return self.y.shape
    
    def __getitem__(self, idx):
        """Allow indexing directly on the Density object."""
        return self.y[idx]
    
    def get(self, normalize=True, as_prod=True):
        """
        Return the (possibly normalized) log-density.
         - normalize indicates whether to normalize the density so it integrates to 1; 
         - as_prod indicates whether to reduce multivariate densities to a single log-density (sum over
           features)
        """
        dens = self.y
        
        # Normalize the density using trapezoidal integration
        if normalize:
            if self.x is None:
                dens -= np.trapz(dens, axis=0)
            else:
                dens -= np.trapz(dens, self.x, axis=0)
        
        # Reduce multivariate log-densities to a single log-density if needed
        if as_prod:
            if dens.ndim > 1:
                dens = np.sum(dens, axis=1)
        
        # Return the adjusted log-density, including normalization constant and sample size
        return np.log(self.nc) + dens - np.log(self.n_points)
    
    def cdf(self, x):
        """Compute the cumulative distribution function (CDF) up to a given point."""
        d = len(x)                              # Number of features
        low = np.min(self.x,  axis=0)           # Minimum of support per feature
        support = np.linspace(low, x, num=100)  # Generate 100 points from min to x
        dens = np.exp(self.y)                   # Convert log-density back to density

        # Integrate density along each feature using trapezoidal rule. It is an array of length d,
        # containing a separate CDF for each feature.
        cumulative = np.array([np.trapz(dens, support[:, i]) for i in range(d)], dtype=np.float64)

        return cumulative
