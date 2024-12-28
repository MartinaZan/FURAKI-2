import numpy as np
from ..stats.base import StatsTracker

class Interval:
    def __init__(self, a=None, b=None) -> None:     
        self.inf = a
        self.sup = b

    def update(self, x):
        if self.inf is None:
            self.inf = x
        else:
            self.inf = min(x, self.inf)
        
        if self.sup is None:
            self.sup = x
        else:
            self.sup = max(x, self.sup)

class Grid:
    def __init__(self):
        self.size = 0
        self.length = 0
        self.portions = []

    
    @classmethod
    def build_from(cls, X):
        gd = cls()
        gd.length = len(X) if X.ndim > 1 else 1
        gd.size = X.shape[1] if X.ndim > 1 else len(X)
        gd.portions.clear()
        minis = X.min(0)
        maxis = X.max(0)
        for j in range(gd.size):
            a = minis[j]
            b = maxis[j]
            gd.portions.append(Interval(a, b))
        return gd

    def update(self, subject: StatsTracker):
            self.length += 1
            x = subject._state

            if self.size < len(x):
                for _ in range(len(x) - self.size):
                    self.portions.append(Interval())
                self.size = len(x)
                                     
            for j in range(self.size):
                self.portions[j].update(x[j])
    
    def get(self, size=None):
        l = list(map(lambda i: i.inf, self.portions))
        u = list(map(lambda i: i.sup, self.portions))
        return np.linspace(l, u, num=self.length if size is None else size)

    def __iter__(self):
        return (self.portions[i] for i in range(self.size))
    
    def reset(self):
        self.length = 0
        self.portions = [Interval() for _ in range(self.size)]