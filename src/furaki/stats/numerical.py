import numpy as np
from ..stats.base import StatsTracker
from ..kernels import make_kernel, Bandwidth, Grid

class CovarianceTracker(StatsTracker):
    is_numeric = True

    def __init__(self, numerical_features=None, **kwargs) -> None:
        super().__init__(numerical_features)
        kernel_method = kwargs.get("kernel", "gaussian")
        bw_method = kwargs.get("bw_method", "scott")
        threshold_crit = kwargs.get("threshold_criterion", 'midpoint')
        bw = Bandwidth.create(bw_method)
        self.gd = Grid()
        self.kernel = make_kernel(kernel_method, bw, self.gd)
        
        self.attach(bw)
        self.attach(self.gd)
        self.attach(self.kernel)
        
        self.threshold_criterion = threshold_crit
        self.mean=0
        self.pvar=0
        self.pcov=0
        self.suggest_split_where = 0


    def update(self, x):
        x = x[self.feature_names].to_numpy()
        self._state = x
        
        if self.size is None:
            self.size = len(x)
            # self.lsupport = np.array([np.inf] * self.size)
            # self.hsupport = np.array([-np.inf] * self.size)
        
        self.n_points += 1
        
        delta = x - self.mean
        self.mean += delta / self.n_points
        delta_at_n = (x - self.mean) / self.n_points
        delta2 = x - self.mean
        self.pvar += delta * delta2

        shp = (self.size, self.size)
        D_at_n = (delta * np.identity(self.size)).dot(np.broadcast_to(delta_at_n, shp))
        self.pcov = self.pcov * ((self.n_points - 1) / self.n_points) + D_at_n if self.n_points > 1 else 0

        
        # self.pvar[self.pvar < 0] = 1e-12

        #self.klls.update(x)
        
        
        # self.kernel.update(x)
        # self.update_kernel(x, self.getmean())
        self.notify()
    
    def get_threshold(self, attribute):
        
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
        return np.array(self.mean, dtype=np.float128)
    
    def getvar(self, as_sum=False):
        return np.zeros((self.size,), dtype=np.float128) if self.n_points < 2 \
            else np.array(self.pvar / (self.n_points - 1), dtype=np.float128)

    def getcov(self):
        return np.zeros((self.size, self.size), dtype=np.float128) if self.n_points < 2 \
            else np.array(self.pcov / (self.n_points - 1) * self.n_points, dtype=np.float128)
    
    def getpdf(self, support=None):
        return self.kernel.estimate(support)
    
    def reset(self):
        super().reset()
        self.mean = 0
        self.pvar = 0
        self.pcov = 0
        self.gd.reset()
        self.kernel.reset_kernel()
