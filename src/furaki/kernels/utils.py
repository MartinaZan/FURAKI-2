import numpy as np
from scipy.signal import correlate
from .cosine import Cosine
from .epanechnikov import Epanechnikov
from .gaussian import Gaussian
from ..kernels import Kernel, Density

def pdfsum(ks: list[Kernel], support=None):

    z = []
    zs = []
    for i in range(len(ks)):
        zi = ks[i].estimate() if support is None else ks[i].estimate(support[i])
        z.append(zi.get(normalize=True, as_prod=True))
        zs.append(zi.x)
    
    observed_density = correlate(np.exp(z[0]), np.exp(z[1]), mode='same') if len(ks) > 1 else np.exp(z[0])
    observed_support = np.linspace(
        np.concatenate([zs[0].min(0), zs[1].min(0)], axis=0) if len(ks) > 1 else zs[0].min(0),
        np.concatenate([zs[0].max(0), zs[1].max(0)], axis=0) if len(ks) > 1 else zs[0].max(0),
        100)
    pdf = Density(observed_density, observed_support)

    return pdf


def normalize(x):
    return (x - min(x))/(max(x)-min(x))

def get_trackers_mean(stats_trackers):
    means = []
    for tracker in stats_trackers:
        means.extend(tracker.getmean())
    return np.array(means).ravel()


def make_kernel(name, bw, grid):
    match name:
        case "gaussian":
            return Gaussian(bw, grid)
        case "cosine":
            return Cosine(bw, grid)
        case "epanechnikov":
            return Epanechnikov(bw, grid)
        case _:
            raise NotImplementedError()
