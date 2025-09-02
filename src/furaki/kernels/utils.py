import numpy as np
from scipy.signal import correlate
from .cosine import Cosine
from .epanechnikov import Epanechnikov
from .gaussian import Gaussian
from ..kernels import Kernel, Density

def pdfsum(ks: list[Kernel], support=None):
    """Combine multiple kernel density estimates (KDEs) into a single density."""

    z = []      # List to store log-density arrays from each kernel
    zs = []     # List to store corresponding support grids

    for i in range(len(ks)):
        # Get the current kernel's density estimate
        zi = ks[i].estimate() if support is None else ks[i].estimate(support[i])
        z.append(zi.get(normalize=True, as_prod=True))  # normalized log-density
        zs.append(zi.x)                                 # store the support grid
    
    # If more than one kernel, combine densities using cross-correlation. Otherwise, use the
    # first kernel's density
    observed_density = correlate(np.exp(z[0]), np.exp(z[1]), mode='same') if len(ks) > 1 else np.exp(z[0])

    # Construct the support for the combined density
    observed_support = np.linspace(
        np.concatenate([zs[0].min(0), zs[1].min(0)], axis=0) if len(ks) > 1 else zs[0].min(0),
        np.concatenate([zs[0].max(0), zs[1].max(0)], axis=0) if len(ks) > 1 else zs[0].max(0),
        100)
    
    # Wrap the resulting density and support into a Density object. This is a single density.
    pdf = Density(observed_density, observed_support)       # NOTE: this is not clear to me

    return pdf


def normalize(x):
    """Normalization to the range [0, 1]."""
    return (x - min(x))/(max(x)-min(x))

def get_trackers_mean(stats_trackers):
    """Compute the combined mean values from a list of StatsTracker objects."""
    means = []
    for tracker in stats_trackers:
        means.extend(tracker.getmean())
    return np.array(means).ravel()


def make_kernel(name, bw, grid):
    """Factory function to create a kernel object based on its name."""
    match name:
        case "gaussian":
            return Gaussian(bw, grid)
        case "cosine":
            return Cosine(bw, grid)
        case "epanechnikov":
            return Epanechnikov(bw, grid)
        case _:
            raise NotImplementedError()
