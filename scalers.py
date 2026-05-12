import numpy as np


class MapMinMax:
    def __init__(self,remove_constant=True):
        self.remove_constant=remove_constant
        self.mask=None
        self.xmin=None
        self.scale=None

    def fit(self,X):
        xmin=X.min(axis=0)
        xmax=X.max(axis=0)
        if self.remove_constant:
            mask=(xmax>xmin)
        else:
            mask=np.ones_like(xmax,dtype=bool)
        self.mask=mask
        self.xmin=xmin[mask]
        rng=xmax[mask]-xmin[mask]
        rng_safe=np.where(rng==0,1.0,rng)
        self.scale=2.0/rng_safe
        return self

    def transform(self,X):
        X2=X[:,self.mask]
        Y=X2-self.xmin
        Y=Y*self.scale
        Y=Y-1.0
        return Y
