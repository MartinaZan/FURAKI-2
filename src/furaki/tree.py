from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple, Union
from natsort import natsorted

from itertools import permutations
import pandas as pd
import numpy as np
import igraph as ig

from sklearn.base import ClusterMixin
from sklearn.tree import BaseDecisionTree
from sklearn.utils.validation import (
    check_is_fitted,
)
from .incremental_tree import IncrementalTree

class FurakiTree(ClusterMixin, BaseDecisionTree):

    def __init__(
        self,
        *,
        criterion="infogain",
        splitter="g-test",
        window='standard',
        grid_size=80,
        reset_time=1.0,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,

        # start furaki args
        # common tracker params:
        # kernel
        # gridsize
        # baseline_wait
        # bw_method
        numerical_tracker="covariance",
        kernel='gaussian',
        bandwidth='scott',
        categorical_tracker="frequency",
        threshold_criterion="midpoint",
        sample_test_size=150,
        alpha=0.95,
        **kwargs
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            monotonic_cst=monotonic_cst,
            ccp_alpha=ccp_alpha,
        )

        # start furaki params
        
        self.criterion = criterion
        self.splitter = splitter
        self.window = window
        self.grid_size = grid_size
        self.reset_time = reset_time
        self.max_depth = max_depth
        self.numerical_tracker = numerical_tracker
        self.categorical_tracker = categorical_tracker
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.threshold_criterion = threshold_criterion
        self.sample_test_size = sample_test_size
        self.alpha = alpha
        
        # override BaseDecisionTree
        self.incremental_tree = IncrementalTree(
            criterion=self.criterion,
            splitter=self.splitter,
            window=self.window,
            max_depth=max_depth,

            numerical_tracker=self.numerical_tracker,
            numerical_tracker_params=dict(kernel=self.kernel, bw_method=self.bandwidth),
            window_params=dict(gridsize=self.grid_size, reset_time=self.reset_time),
            categorical_tracker=self.categorical_tracker,
            categorical_tracker_params=dict(kernel=self.kernel, bw_method=self.bandwidth),
            baseline_wait=self.sample_test_size,
            sample_test_size=self.sample_test_size,
            alpha=self.alpha
        )
        
       
        self.labels = []
        

        self._prune_candidates = set()
        self._merge_candidates = set()

        # this attribute will get created after the first time
        # fit is called on the tree
        # self.incremental_tree:Tree = None

    def __sklearn_is_fitted__(self):
        return self.incremental_tree.get_root() is not None

    """Properties"""
    @property
    def count(self) -> int:
        s = 0
        for node in self.all_nodes():
            s += node.count
        return s
    
    @property
    def num_splits(self):
        return self.incremental_tree.num_splits

    #Override
    @property
    def children_left(self):
        """children_left[i]: id of the left child of node i or -1 if leaf node"""

    #Override
    @property
    def children_right(self):
        """children_right[i]: id of the right child of node i or -1 if leaf node"""

    # dtreeviz/Override
    @property
    def threshold(self):
        """threshold[i]: threshold value at node i"""
        check_is_fitted(self)

    #dtreeviz/Override
    @property
    def feature(self):
        """feature is an array where feature[i] describes feature used to split node i"""
        check_is_fitted(self)

    # Override/dtreeviz
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        return self.incremental_tree.compute_feature_importances()

    @property
    def n_features(self) -> int:
        return len(self._numerical_features) + len(self._categorical_features)
    
    @property
    def n_classes_(self):
        return len(self.leaves())

    """Public methods"""

    def fit(self, X, y=None, sample_weight=None, check_input=True):
        
        # Determine output settings
        # n_samples, self.n_features_in_ = X.shape
        self.incremental_tree = IncrementalTree(
            criterion=self.criterion,
            splitter=self.splitter,
            window=self.window,
            max_depth=None,

            numerical_tracker=self.numerical_tracker,
            numerical_tracker_params=dict(kernel=self.kernel, bw_method=self.bandwidth),
            window_params=dict(gridsize=self.grid_size, reset_time=self.reset_time),
            categorical_tracker=self.categorical_tracker,
            categorical_tracker_params=dict(kernel=self.kernel, bw_method=self.bandwidth),
            baseline_wait=self.sample_test_size,
            sample_test_size=self.sample_test_size,
            alpha=self.alpha
        )
        self.labels = []
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        
        self.feature_names_in_ = X.columns
        self.incremental_tree._extract_features(X)
        # Increase tree
        L = X.apply(self.incremental_tree._learn_one, axis=1)
        
        self.labels = L

        return self
    
    # Override
    def apply(self, X:pd.DataFrame, y=None):
        """"""
        check_is_fitted(self)
        return self.incremental_tree._apply(X)
    
    def predict(self):
        """
        """
        L = []
        for node_id in self.labels:
            L.append(self.get_node(node_id).label)
        return np.array(L)
    
    def predict_drift(self, X):
        L = self.predict(X)
        Ls = ''.join([str(v) for v in L])
        symbols = natsorted(list(map(str, set(L))))
        counter = {}
        keys = [p for p in permutations(symbols, 3) if p[0] < p[1] < p[2]]
        counter.update({p:0 for p in keys})
        for k in keys:
            counter.update({k: Ls.count(k[0]+k[1]+k[2])})

        return counter
            

    def predict_count(self, X):
        L = []
        for node_id in self.labels:
            L.append(self.get_node(node_id).count)
        return np.array(L)
    
    def predict_node_type(self, X):
        L = []
        for node_id in self.labels:
            L.append(self.get_node(node_id).is_leaf())
        return np.array(L)
    
    def decision_path(self, X, check_input=True):
        pass
        
    def predict_proba(self, X, check_input=True):
        pass
    def predict_log_proba(self, X):
        pass

    def score(self, X, y=None):
        pass



    
    def get_leaves(self, filter_func=None):
        return list(
            filter(filter_func, self.incremental_tree.leaves())
        )
    
    def get_nodes(self, filter_func=None):
        return list(
            filter(filter_func, self.incremental_tree.all_nodes())
        )

    # Override BaseDecisionTree
    def get_n_leaves(self):
        check_is_fitted(self)
        return len(self.incremental_tree.leaves())
    
    def get_depth(self):
        check_is_fitted(self)
        return self.incremental_tree.depth() # get max level of the entire tree


    # EXPORT

    def to_networkx(self):
        """Export the tree in networkx data format"""
        G = ig.Graph(directed=True)
        nr_vertices = len(self.get_nodes())
        # ids = list(map(lambda node: node.identifier, self.get_nodes()))
        # mapping = {k: ids[k] for k in range(nr_vertices)}
        # G.add_vertices(ids)
        nodes_size = []
        nodes_deviance=[]
        queue = [G.add_vertex(shape="record", label="{root|"+f"N: {self.get_root().count}"+"}", ident=self.get_root().identifier,size=float(self.get_root().count), style='filled')]
        i =1
        while queue: # gather all nodes, leaves will not be added by the loop
            e = queue.pop(0)
            node = self.get_node(e['ident'])
            if node is not None:
                #e = G.add_vertex(name=node.identifier)
                if node.left_child is not None:
                        
                    l1 = G.add_vertex(shape="record", label="{"+f"id: {node.left_child.label}"+f" | N: {node.left_child.count}"+"}", ident=node.left_child.identifier, style='filled', size=float(node.left_child.count))
                    if node.attribute in list(map(lambda n: n[1], node._categorical_features)): 
                        mode="=" 
                    else: mode="<="
                    G.add_edges(
                        [(e, l1)], 
                        dict(label=f"{node.attribute} {mode} {str(node.threshold)[:6] if '.' in str(node.threshold) else node.threshold}"))
                    queue.append(l1)
                if node.right_child is not None:
                    r1 = G.add_vertex(shape="record", label="{"+f"id: {node.right_child.label}"+f" | N: {node.right_child.count}"+"}", ident=node.right_child.identifier, style='filled', size=float(node.right_child.count))
                    # if node.attribute in list(map(lambda n: n[1], node._categorical_features)): 
                    #     mode="!=" 
                    # else: mode=">"
                    G.add_edges(
                        [(e, r1)],
                        dict(label=""))
                        #dict(label=f"{node.attribute} {mode} {node.threshold}"))
                    queue.append(r1)
            i+=1
        # Store layout as (xn, yn) into each node and (xe, ye) into each edge
        # layout = buchheim(self)
        

        graph = G.to_networkx()
        
        return graph

    """Public methods"""


    """Private methods"""
    
    # def _extract_features(self, X:pd.DataFrame):
        # self._numerical_features = []
        # self._categorical_features = []
        # for i, dt in enumerate(X.dtypes):49403tr(dt): # value dt is an object, possibly string
        #         self._categorical_features.append((i, X.columns[i]))


    def get_root(self):
        return self.incremental_tree.get_root()
    
    def get_node(self, identifier):
        return self.incremental_tree.get_node(identifier)
