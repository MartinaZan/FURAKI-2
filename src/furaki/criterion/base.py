from abc import abstractmethod
import numpy as np

class Criterion():

    def __call__(self, o, e):
        infogain = self._criterion(o, e)
        feature_index = np.argmax(infogain)
        
        return infogain[feature_index], feature_index
    
    @abstractmethod
    def _criterion(self, H, Ha):
        pass
    

class IGCriterion(Criterion):

    def _criterion(self, H, Ha):
        return H - Ha
    
class IGRatioCriterion(Criterion):

    def _criterion(self, H, Ha):
        return Ha / H