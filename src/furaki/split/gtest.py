import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import chi2
from ..split import Splitter

class GTestKLDSplit(Splitter):
    def __init__(self, *stats_trackers, alpha=0.95, sample_test_size=150, criterion=None):
        self.alpha = alpha
        self.stats_trackers = [st for st in stats_trackers if st is not None]
        self.k = np.sum([t.n_features for t in self.stats_trackers])
        self.criterion = criterion
        self.expected = None
        self.support = None
        self.reference = None
        self.counter = 0
        self.sample_test_size = sample_test_size

        self.split_feature_index = -1
        self.split_feature_value = -1

    def update(self, x):
        pass
        

    def check(self, observed, reference, w):
        
        c = observed.get(normalize=False, as_prod=False)
        r = reference.get(normalize=False, as_prod=False)
        #k = np.sum([t.n_features for t in self.stats_trackers])
        cross_ent = -trapezoid(np.exp(r) * c)
        entropy = -trapezoid(np.exp(r) * r)
        # print(cross_ent, entropy, cross_ent - entropy)
        kld = max(0, cross_ent - entropy) # kld = cross_ent - ent + 1e-3
        # if np.isnan(kld) or np.isclose(abs(kld), 0, atol=1e-12):
        #     kld = 0
        # assert kld >= 0, f"KLD should be >= 0 instead is {kld}"
        G = float(2 * 100 * kld)
        pvalue = chi2.sf(G, df=self.k-1)
        return pvalue <= (1 - self.alpha)