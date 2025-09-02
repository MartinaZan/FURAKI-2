import numpy as np
from ..stats.base import StatsTracker
from ..kernels import make_kernel, Bandwidth, Grid

class CovarianceTracker(StatsTracker):
    is_numeric = True   # This tracker is for numerical features only

    def __init__(self, numerical_features=None, **kwargs) -> None:
        super().__init__(numerical_features)

        # Configuration parameters
        kernel_method = kwargs.get("kernel", "gaussian")
        bw_method = kwargs.get("bw_method", "scott")
        threshold_crit = kwargs.get("threshold_criterion", 'midpoint')
        
        # Create bandwidth, grid, and kernel objects
        bw = Bandwidth.create(bw_method)
        self.gd = Grid()
        self.kernel = make_kernel(kernel_method, bw, self.gd)
        
        # Register them as observers
        self.attach(bw)
        self.attach(self.gd)
        self.attach(self.kernel)
        
        self.threshold_criterion = threshold_crit
        self.mean = 0
        self.pvar = 0
        self.pcov = 0
        self.suggest_split_where = 0


    def update(self, x):
        """
        Update the tracker with a new sample x.
        """
        x = x[self.feature_names].to_numpy()
        self._state = x
        
        # Initialize size if not already set
        if self.size is None:
            self.size = len(x)
        
        # Increment number of observed points
        self.n_points += 1
        
        # Update mean and variance online
        delta = x - self.mean
        self.mean += delta / self.n_points
        delta_at_n = (x - self.mean) / self.n_points
        delta2 = x - self.mean
        self.pvar += delta * delta2

        # Update covariance matrix
        shp = (self.size, self.size)
        D_at_n = (delta * np.identity(self.size)).dot(np.broadcast_to(delta_at_n, shp))
        self.pcov = self.pcov * ((self.n_points - 1) / self.n_points) + D_at_n if self.n_points > 1 else 0

        # Notify observers
        self.notify()
    
    def get_threshold(self, attribute):
        """Suggest a split threshold for a given feature."""
        match self.threshold_criterion:
            case "highest_avg":
                pass
            case "midpoint":
                suggest_split_where = self.getmean()
            
        idx = self.feature_indices.index(attribute)
        
        try:
            return suggest_split_where[idx]
        except:
            return 0

    def getmean(self, as_sum=False):
        """Get the mean of tracked features."""
        return np.array(self.mean, dtype=np.longdouble)
    
    def getvar(self, as_sum=False):
        """Get the variance of tracked features."""
        return np.zeros((self.size,), dtype=np.longdouble) if self.n_points < 2 \
            else np.array(self.pvar / (self.n_points - 1), dtype=np.longdouble)

    def getcov(self):
        """Get the covariance matrix of tracked features."""
        return np.zeros((self.size, self.size), dtype=np.longdouble) if self.n_points < 2 \
            else np.array(self.pcov / (self.n_points - 1) * self.n_points, dtype=np.longdouble)
    
    def getpdf(self, support=None):
        """Estimate the probability density function using the kernel."""
        return self.kernel.estimate(support)
    
    def reset(self):
        """Reset all internal statistics and observers to initial state."""
        super().reset()
        self.mean = 0
        self.pvar = 0
        self.pcov = 0
        self.gd.reset()
        self.kernel.reset_kernel()
