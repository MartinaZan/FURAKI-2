from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from ..base import Subject
from ..stats.base import StatsTracker

class Bandwidth:
    """Used to calculate the bandwidth for kernels."""

    @staticmethod
    def create(name):
        match name:
            case "scott":
                return Scott()
            case "silverman":
                return Silverman()
            case "variance":
                return Variance()
            case _:
                raise NotImplementedError("use 'scott' or 'silverman'")

    def __init__(self, name) -> None:
        self.name = name
        self.value = 0

    @abstractmethod
    def update(self, subject: Subject):
        pass

    def get(self):
        return self.value

class Scott(Bandwidth):
    """Scott: uses Scott's rule (1.059 * std * n^(-1/(4+d)))"""

    def __init__(self) -> None:
        super().__init__("scott")

    def update(self, subject: StatsTracker):
        var = subject.getvar()
        var[var < 0] = 0
        std = np.sqrt(var)
        n = subject.n_points
        d = len(std)
        scotts_factor = n**(-1./(4+d))
        value = 1.059 * std * scotts_factor
        value[value <= 0.] = 1.
        self.value = value

class Silverman(Bandwidth):
    """Silverman: uses Silverman's rule (approximation using std and IQR)"""

    def __init__(self) -> None:
        super().__init__("silverman")

    def update(self, subject: StatsTracker):
        var = subject.getvar()
        var[var < 0] = 0
        std = np.sqrt(var)
        mu = subject.getmean()
        n = subject.n_points
        d = len(std)
        q = (
            mu - 0.675 * std,
            mu + 0.675 * std
        )
        IQR = q[1] - q[0]
        dispersion = np.minimum(std, IQR / 1.349)
        scotts_factor = n**(-1./(4+d))
        value = .9 * dispersion * scotts_factor
        value[value <= 0.] = 1.
        self.value = value

class Variance(Bandwidth):
    """Variance: uses the variance of each feature as bandwidth."""
    def __init__(self) -> None:
        super().__init__("variance")

    def update(self, subject: StatsTracker):
        value = subject.getvar()
        value[value == 0.] = 1.
        self.value = value