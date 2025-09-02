from abc import abstractmethod
import numpy as np

class Criterion():

    def __call__(self, o, e): # o = current, observed; e = reference, expected
        infogain = self._criterion(o, e)        # Compute information gain for each feature
        feature_index = np.argmax(infogain)     # Select the feature with the maximum information gain
        
        return infogain[feature_index], feature_index
    
    @abstractmethod
    def _criterion(self, H, Ha):
        pass
    

class IGCriterion(Criterion):
    def _criterion(self, H, Ha):
        return H - Ha               # Non mi è chiaro se devo fare H - Ha o Ha - H


class IGRatioCriterion(Criterion):  # NOTE: is this correct?
    def _criterion(self, H, Ha):
        return Ha / H