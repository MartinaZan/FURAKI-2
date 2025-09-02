import numpy as np
from collections import defaultdict
from ..stats.base import StatsTracker
from ..kernels import make_kernel, Bandwidth, Grid

class FrequencyTracker(StatsTracker):
    is_numeric = False  # This tracker is for categorical features only

    def __init__(self, categorical_features=None, **kwargs):
        super().__init__(categorical_features)

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
        self.size = None
        self.categories = []
        self.num_categories = 0
        self.prev_categories = 0

        self.ewjk = {k:defaultdict(int) for k in self.feature_names}
        self.suggest_split_where = 0

    @staticmethod
    def make_diagonal(diag_fn, off_fn, arr):
        """Builds a square matrix with diag_fn on the diagonal and off_fn elsewhere"""
        D = np.zeros((len(arr), len(arr)))
        for i in range(len(arr)):
            for j in range(len(arr)):
                if i==j:
                    D[i,j] = diag_fn(arr[i], arr[j])
                else:
                    D[i,j] = off_fn(arr[i], arr[j])
        return D


    def update(self, x):
        """
        Update frequency tables with a new sample x. Each feature's categorical value count is incremented.
        """
        x = x[self.feature_names]
        self.n_points += 1

        # Update counts for each categorical feature
        for j, c in enumerate(self.feature_names):
            self.ewjk[c][x[c]] += 1
        
        # Update size and categories
        self.size = [len(self.ewjk[c].keys()) for c in self.feature_names]
        cat = []
        for c in self.feature_names:
            cat.append(list(self.ewjk[c].keys()))
        self.categories = cat
        
        # Track unique categories across all features
        ret = set()
        for n in range(self.n_features):
            for c in self.categories[n]:
                ret.add(c)
        self.num_categories = len(ret)
        self.unique_categories = ret

        # Store current state as a vectorized representation and notify observers
        self._state = self.vectorize(x)
        self.notify()

    def vectorize(self, x=None):
        """
        Convert frequencies into a vector form.
        If x is given, return frequency vector for that specific observation.
        Otherwise, return flattened frequency array.
        """
        if x is not None:
            u = []
            freq = self.getfreq()
            for c in self.feature_names:
                u.append(freq[c].get(x[c], 0.0))
            return np.array(u)

        v=self.getarrfreq(as_vec=True)
        return np.array(v)

    def getcounts(self):
        """Return raw counts for each category of each feature"""
        return {c:{k:self.ewjk[c][k] for k in self.ewjk[c].keys()} for c in self.feature_names}

    def getarrcounts(self, as_vec=False, as_sum=False):
        """
        Return counts as arrays.
        - as_vec=True → flat concatenated vector
        - as_sum=True → one sum per feature
        """
        counts = self.getcounts()
        if as_vec:
            return np.concatenate([list(counts[c].values()) for c in self.feature_names], axis=0)
        indices = np.cumsum(self.size)[:-1]
        n = np.array_split(
            np.concatenate([list(counts[c].values()) for c in self.feature_names], axis=0),
            indices,
            axis=0
        )
        if as_sum:
            return [sum(ns) for ns in n]
        return n

    
    def getfreq(self):
        """Return relative frequencies for each category of each feature"""
        return {c:{k:self.ewjk[c][k]/(self.n_points * self.n_features) for k in self.ewjk[c].keys()} for c in self.feature_names}
    
    def getarrfreq(self, as_vec=False, as_sum=True):
        """
        Return frequencies as arrays.
        - as_vec=True → flat concatenated vector
        - as_sum=True → one sum per feature
        """
        freq = self.getfreq()
        if as_vec:
            return np.concatenate([list(freq[c].values()) for c in self.feature_names], axis=0)
        indices = np.cumsum(self.size)[:-1]
        f = np.array_split(
            np.hstack([list(freq[c].values()) for c in self.feature_names]),
            indices, 
            axis=0)
        if as_sum:
            return [sum(fs) for fs in f]
        return f

    def getmean(self, as_sum=False):
        """
        Return mean category counts (based on frequencies per sample size).
        If as_sum=True, aggregate counts across categories to get one value per feature; 
        if False, return counts for each category individually.
        """
        pi = self.getarrfreq(as_vec=False, as_sum=True)
        mean = self.n_features * self.n_points * np.array(pi)   # expected number of occurrences
        
        # If we want the total expected number of occurrences per feature, instead of per category.
        if as_sum:
            indices = np.cumsum(self.size)[:-1]
            mean = np.array_split(mean, indices, axis=0)
            return np.hstack([sum(m) for m in mean])
        return np.squeeze(mean)
    
    def getvar(self, as_sum=False):
        """
        Return variance for each categorical feature.
        If as_sum=True, aggregate counts across categories to get one value per feature; 
        if False, return counts for each category individually.
        """
        pi = np.array(self.getarrfreq(as_vec=False, as_sum=True))
        var = self.n_features * self.n_points * pi * (1 - pi)

        # If we want the total expected number of occurrences per feature, instead of per category.
        if as_sum:
            indices = np.cumsum(self.size)[:-1]
            var = np.array_split(var, indices, axis=0)
            return np.hstack([sum(v) for v in var])
        return np.squeeze(var)

    def getcov(self, type='covariance'):
        """
        Return covariance matrix between categories.
        - 'covariance': full covariance (negative off-diagonals)
        - 'diagonal': only variances
        """
        pi = self.getarrfreq(as_vec=True)
        if type=="covariance":
            off_fn = lambda pii, pij: -self.n_features*self.n_points*pii*pij
        if type=="diagonal":
            off_fn = lambda pii, pij: 0.
        return self.make_diagonal(lambda pii, pij: self.n_features * self.n_points * pii * (1 - pij), off_fn , pi)
    
    def getpdf(self, support=None):
        """Return probability density estimation (via kernel)"""
        return self.kernel.estimate(support)
    
    def get_threshold(self, attribute):
        """
        Return threshold for splitting based on criterion.
        Currently only 'midpoint' supported → uses mean frequency.
        """
        match self.threshold_criterion:
            case "highest_avg":
                raise NotImplementedError("use 'midpoint'")
            case "midpoint":
                suggest_split_where = self.getmean()
            
        idx = self.feature_indices.index(attribute)

        try:
            return suggest_split_where[idx]
        except:
            return 0

    def reset(self):
        """Reset all counts and related trackers"""
        for c in self.feature_names:
            for k in self.ewjk[c].keys():
                self.ewjk[c][k] = 0
        self.gd.reset()
        self.kernel.reset_kernel()