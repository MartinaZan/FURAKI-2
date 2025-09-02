import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import chi2
from ..split import Splitter

class GTestKLDSplit(Splitter):
    """
    Splitter class that uses a G-test based on the Kullback-Leibler divergence (KLD)
    between observed and reference densities to decide whether to split.
    """
    def __init__(self, *stats_trackers, alpha=0.95, sample_test_size=150, criterion=None):
        self.alpha = alpha          # Significance level for the G-test 
        self.stats_trackers = [st for st in stats_trackers if st is not None]
        self.k = np.sum([t.n_features for t in self.stats_trackers])    # Total number of features across all trackers
        self.criterion = criterion

        self.expected = None
        self.support = None
        self.reference = None

        self.counter = 0
        self.sample_test_size = sample_test_size

        # Track the best split feature and value
        self.split_feature_index = -1
        self.split_feature_value = -1

    def update(self, x):
        pass
        

    def check(self, observed, reference, w):
        """
        Perform a G-test to check if the observed distribution differs significantly from
        the reference distribution using the Kullback-Leibler divergence (KLD). It returns
        True if the null hypothesis is rejected (distributions differ), False otherwise.
        """
        # Get the raw (non-normalized) densities for observed and reference
        c = observed.get(normalize=False, as_prod=False)
        r = reference.get(normalize=False, as_prod=False)
        
        # Compute cross-entropy between reference and observed
        cross_ent = -trapezoid(np.exp(r) * c)
        # Compute entropy of reference distribution
        entropy = -trapezoid(np.exp(r) * r)

        # Compute Kullback-Leibler divergence (max to ensure non-negativity despite numerical errors)
        kld = max(0, cross_ent - entropy) # kld = cross_ent - entropy + 1e-3
        
        # Compute G statistic for the G-test
        G = float(2 * 100 * kld)

        # Compute p-value from chi-squared distribution
        pvalue = chi2.sf(G, df=self.k-1)

        # Return True if null hypothesis is rejected (p-value small)
        return pvalue <= (1 - self.alpha)