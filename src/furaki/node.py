from __future__ import annotations
from typing import List, Dict, Tuple, Union
import numpy as np
from functools import partial
from treelib import Node
from treelib.exceptions import *

from .stats.numerical import CovarianceTracker
from .stats.categorical import FrequencyTracker
from .split import GTestKLDSplit
from .criterion import IGCriterion, IGRatioCriterion
from .windows import Recent, Standard

from .stats.base import *

class IncrementalTreeNode(Node):

    def __init__(
        self,
        tag=None,
        identifier=None,
        data=None,
    ):
        super().__init__(tag, identifier, False, data)

        numerical = data["numerical_features"]
        categorical = data["categorical_features"]
        max_depth = data["max_depth"]
        alpha = data["alpha"]
        splitter = data["splitter"]
        criterion = data["criterion"]
        window = data['window']
        min_samples_split = data["min_samples_split"]
        sample_test_size = data["sample_test_size"]
        self.tree_ = data["tree"]

        self.numerical_class = data["numerical"]
        
        self._num_tracker = None
        self._cat_tracker = None
        self._num_class = None
        self._cat_class = None

        self.ft_img = ""


        if len(numerical) > 0:
            self._num_tracker = CovarianceTracker(numerical, **data.get("numerical_params",None))
            self._num_class = partial(CovarianceTracker, numerical, **data.get("numerical_params", None))
        if len(categorical) > 0:
            self._cat_tracker = FrequencyTracker(categorical, **data.get("categorical_params",None))
            self._cat_class = partial(FrequencyTracker, categorical, **data.get("categorical_params",None))

        if isinstance(criterion, str):
            match criterion:
                case "infogain":
                    self._criterion = IGCriterion()
                case 'infogain_ratio':
                    self._criterion = IGRatioCriterion()
                case _:
                    raise NotImplementedError("Unknown criterion specified")

        if isinstance(splitter, str):
            match splitter:
                case 'g-test':
                    self._splitter = GTestKLDSplit(self._num_tracker, self._cat_tracker, alpha=alpha, sample_test_size=sample_test_size, criterion=self._criterion)
                case _:
                    raise NotImplementedError("Unknown splitter specified")
                
        
        if isinstance(window, str):
            match window:
                case 'recent':
                    self._window = Recent(self._num_class, self._cat_class, sample_test_size, **data.get("window_params",None))
                case 'standard':
                    self._window = Standard(self._num_class, self._cat_class, sample_test_size, **data.get("window_params",None))
                case _:
                    raise NotImplementedError("Unknown window specified")
        
        self._numerical_features = numerical
        self._categorical_features = categorical

        self.max_depth = max_depth
        self._min_samples_split = min_samples_split
        
        self._is_learning_disabled = False    # can the node update its summary statistics?
        self._is_growth_disabled = False      # can the node produce two leaf children, a.k.a split?
        self.n_points = 0                      # incremental counter for samples seen by the node
        
        self._left_split_proba = 0
        self._right_split_proba = 0
        self._is_active = True                # computed as function of is_learning_disabled and is_growth_disabled


    """Overridden methods"""
    def __len__(self):
        return self.n_points

    def __str__(self):
        return str(self.identifier)

    """Properties"""

    @property
    def count(self):
        return self.n_points

    @property
    def dimensions(self) -> int:
        return sum([tracker.n_features for tracker in self._window.get_ref_trackers() if tracker is not None])
    
    @property
    def features_all(self) -> list:
        f= []
        for tracker in self._window.get_ref_trackers():
            if tracker is not None:
                for fi, fn in zip(tracker.feature_indices, tracker.feature_names):
                    f.append([fi, fn])
        f.sort(key=lambda e: e[0])
        for idx, fi in enumerate(f):
            #print(fi, [t[1] for t in self._numerical_features])
            if fi[1] in [t[1] for t in self._numerical_features]:
                fi.append(lambda i: self._window.get_cur_trackers()[0].getmean()[i-len(self._categorical_features)])
            else:
                fi.append(lambda i: self._window.get_cur_trackers()[1].categories[i])
        return f
        
    @property
    def parent(self):
        try:
            return self.tree_.parent(self.identifier)
        except NodeIDAbsentError:
            return None

    @property
    def depth(self):
        return self.tree_.depth(self.identifier)

    @property
    def left_child(self) -> IncrementalTreeNode:
        children = self.tree_.children(self.identifier)
        if len(children) == 2:
            return children[0]

    @property
    def right_child(self) -> IncrementalTreeNode:
        children = self.tree_.children(self.identifier)
        if len(children) == 2:
            return children[-1]


    """Public methods"""

    def filter_instance_to_leaf(self, X):
        if self.is_leaf():
            return self
        if self._test_attribute(X):
            return self.left_child.filter_instance_to_leaf(X)
        
        else:
            return self.right_child.filter_instance_to_leaf(X)
        

    def split(self, X):
        if not self.is_leaf():
            return False
        if not self._window.can_split: # at least min_samples in node to attempt a split
            return False
        
        o = self._window.current
        e = self._window.reference
        

        check = self._splitter.check(o, e, self._window.size)

        if check:
            if self._num_tracker is not None:
                n = X[self._num_tracker.feature_names].to_numpy()
            else:
                n = None
            if self._cat_tracker is not None:
                c = X[self._cat_tracker.feature_names].to_dict()
                d = self._window.cat_tracker_cur.vectorize(c)
            else:
                d = None
            
            x = n if d is None else np.concatenate([n, d], axis=0)
            proba = self._window.current.cdf(x)
            proba /= 1-proba
            fi = np.argmax(proba)
            
            fii, fna, fn = self.features_all[fi] # get feature name
            self._splitter.split_feature_index = fi
            self._splitter.split_feature_value = fn(fi) if (fi in self._num_tracker.feature_indices or d is None) else list(c.values())[fi]


        self._window.refresh(check)
            
        return check

    def clear(self):
        self._criterion.clear()
        self.n_points = 0


    @property
    def attribute(self):
        fd = {}
        fd.update({idx: fname for idx, fname in self._numerical_features})
        fd.update({idx: fname for idx, fname in self._categorical_features})
        return fd[self._splitter.split_feature_index]

    @property
    def threshold(self):
        if type(self._splitter.split_feature_value) is float:
            return f"{self._splitter.split_feature_value:.4f}"
        else:
            return str(self._splitter.split_feature_value)
        
    
    """Private methods"""

    def _test_attribute(self, X):
        if self._num_tracker is not None:
            n = X[self._num_tracker.feature_names].to_numpy()
        else:
            n = None
        if self._cat_tracker is not None:
            u = []
            c = X[self._cat_tracker.feature_names].to_dict()
            c = self._window.cat_tracker_cur.vectorize(c)
            c = np.array(c)
        else:
            c = None

        x = n if c is None else np.concatenate([n, c], axis=0)
        
        m1, c1 = self._window.frozen_left[0], self._window.frozen_left[1]
        m2, c2 = self._window.frozen_right[0], self._window.frozen_right[1]
        go_left = np.sqrt(max(0, (x-m1) @ c1 @ (x-m1)))
        go_right = np.sqrt(max(0, (x-m2) @ c2 @ (x-m2)))
        
        return np.argmin([go_right, go_left])

    def _learn_one(self, X) -> None:
        self.n_points += 1
        self._window.learn_one(X)
