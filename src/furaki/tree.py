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
        criterion = "infogain",             # Information gain criterion for splits
        splitter = "g-test",                # Statistical test for splitting decisions
        window = 'standard',                # Type of sliding window (recent/standard)
        grid_size = 80,                     # Size of the grid for density estimation
        reset_time = 1.0,                   # Time interval for resetting statistics
        max_depth = None,                   # Maximum depth of the tree
        min_samples_split = 2,              # Minimum samples required to split a node
        min_samples_leaf = 1,               # Minimum samples required at a leaf node
        min_weight_fraction_leaf = 0.0,     # Minimum fraction at a leaf
        max_features = None,                # Maximum features to consider for splits
        random_state = None,                # Random state for reproducibility
        max_leaf_nodes = None,              # Maximum number of leaf nodes
        min_impurity_decrease = 0.0,        # Minimum impurity decrease for splits
        class_weight = None,                # Weights associated with classes
        ccp_alpha = 0.0,                    # Complexity parameter for pruning
        monotonic_cst = None,               # Monotonic constraints

        # start furaki args
        # common tracker params:
        # kernel
        # gridsize
        # baseline_wait
        # bw_method
        numerical_tracker = "covariance",   # Tracker type for numerical features
        kernel = 'gaussian',                # Kernel function for density estimation
        bandwidth = 'scott',                # Bandwidth method for kernel density estimation
        categorical_tracker = "frequency",  # Tracker type for categorical features
        threshold_criterion = "midpoint",   # Criterion for threshold selection
        sample_test_size = 150,             # Size of sample for statistical tests
        alpha = 0.95,                       # Confidence level for statistical tests
        **kwargs
    ):
        super().__init__(
            criterion = criterion,
            splitter = splitter,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            min_weight_fraction_leaf = min_weight_fraction_leaf,
            max_features = max_features,
            max_leaf_nodes = max_leaf_nodes,
            class_weight = class_weight,
            random_state = random_state,
            min_impurity_decrease = min_impurity_decrease,
            monotonic_cst = monotonic_cst,
            ccp_alpha = ccp_alpha,
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
            criterion = self.criterion,
            splitter = self.splitter,
            window = self.window,
            max_depth = max_depth,

            numerical_tracker = self.numerical_tracker,
            numerical_tracker_params = dict(kernel=self.kernel, bw_method=self.bandwidth),
            window_params = dict(gridsize=self.grid_size, reset_time=self.reset_time),
            categorical_tracker = self.categorical_tracker,
            categorical_tracker_params = dict(kernel=self.kernel, bw_method=self.bandwidth),
            baseline_wait = self.sample_test_size,
            sample_test_size = self.sample_test_size,
            alpha = self.alpha
        )
        
        # Store labels assigned to each processed sample
        self.labels = []
        
        # Sets to track potential pruning and merging candidates
        self._prune_candidates = set()
        self._merge_candidates = set()

        # this attribute will get created after the first time
        # fit is called on the tree
        # self.incremental_tree:Tree = None

    def __sklearn_is_fitted__(self):
        """Check if the model has been fitted. Required by scikit-learn interface."""
        return self.incremental_tree.get_root() is not None

    """Properties"""
    @property
    def count(self) -> int:
        """Total number of samples processed by the tree."""
        s = 0
        for node in self.all_nodes():
            s += node.count
        return s
    
    @property
    def num_splits(self):
        """Number of splits performed in the tree."""
        return self.incremental_tree.num_splits

    # Override
    @property
    def children_left(self):
        """children_left[i]: id of the left child of node i or -1 if leaf node"""

    # Override
    @property
    def children_right(self):
        """children_right[i]: id of the right child of node i or -1 if leaf node"""

    # dtreeviz/Override
    @property
    def threshold(self):
        """
        Array of threshold values used for splits
        threshold[i]: threshold value at node i
        """
        check_is_fitted(self)

    #dtreeviz/Override
    @property
    def feature(self):
        """
        Array of features used for splits
        feature[i]: feature index used to split node i
        """
        check_is_fitted(self)

    # Override/dtreeviz
    @property
    def feature_importances_(self):
        """Feature importance scores based on tree structure."""
        check_is_fitted(self)
        return self.incremental_tree.compute_feature_importances()

    @property
    def n_features(self) -> int:
        """Total number of features in the dataset."""
        return len(self._numerical_features) + len(self._categorical_features)
    
    @property
    def n_classes_(self):
        """Number of classes (clusters) in the tree. Equivalent to the number of leaf nodes."""
        return len(self.leaves())

    """Public methods"""

    def fit(self, X, y=None, sample_weight=None, check_input=True):
        """
        Fit the Furaki tree to the data using incremental learning.
        
        Args:
            X: Input features (pandas DataFrame or numpy array)
            y: Target values (ignored in clustering)
            sample_weight: Sample weights (not implemented)
            check_input: Whether to validate input (not implemented)
        
        Returns:
            self: Fitted estimator
        """
        
        # Determine output settings
        # n_samples, self.n_features_in_ = X.shape
        self.incremental_tree = IncrementalTree(
            criterion = self.criterion,
            splitter = self.splitter,
            window = self.window,
            max_depth = None,

            numerical_tracker = self.numerical_tracker,
            numerical_tracker_params = dict(kernel=self.kernel, bw_method=self.bandwidth),
            window_params = dict(gridsize=self.grid_size, reset_time=self.reset_time),
            categorical_tracker = self.categorical_tracker,
            categorical_tracker_params = dict(kernel=self.kernel, bw_method=self.bandwidth),
            baseline_wait = self.sample_test_size,
            sample_test_size = self.sample_test_size,
            alpha = self.alpha
        )
        # Reset labels for new training
        self.labels = []
        
        # Convert input to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        
        # Store feature names for later use
        self.feature_names_in_ = X.columns

        # Extract and categorize features (numerical vs categorical)
        self.incremental_tree._extract_features(X)

        # Process each sample incrementally to build the tree
        # _learn_one updates the tree structure and returns node assignment
        L = X.apply(self.incremental_tree._learn_one, axis=1)
        
        # Store the node assignments for each sample
        self.labels = L

        return self
    
    # Override
    def apply(self, X:pd.DataFrame, y=None):
        """Apply the tree to get node assignments for new data."""
        check_is_fitted(self)
        return self.incremental_tree._apply(X)
    
    def predict(self):
        """Predict cluster labels for the fitted data."""
        L = []
        for node_id in self.labels:
            L.append(self.get_node(node_id).label)
        return np.array(L)
    
    def predict_drift(self, X):
        """
        Predict concept drift by analyzing cluster transition patterns.
        
        This method looks for specific transition patterns between clusters
        that might indicate concept drift in the data stream.

        It returns the count of different transition patterns.
        """
        # Get cluster labels for all processed samples
        L = self.predict(X)

        # Convert labels to string for pattern matching
        Ls = ''.join([str(v) for v in L])

        # Get sorted unique symbols (cluster labels)
        symbols = natsorted(list(map(str, set(L))))

        # Initialize counter for transition patterns
        counter = {}

        # Generate all possible 3-element ascending permutations
        # These represent potential drift patterns (e.g., cluster 1->2->3)
        keys = [p for p in permutations(symbols, 3) if p[0] < p[1] < p[2]]
        counter.update({p:0 for p in keys})

        # Count occurrences of each transition pattern in the label sequence
        for k in keys:
            counter.update({k: Ls.count(k[0]+k[1]+k[2])})

        return counter
            

    def predict_count(self, X):
        """Predict the sample count for each assigned node."""
        L = []
        for node_id in self.labels:
            L.append(self.get_node(node_id).count)
        return np.array(L)
    
    def predict_node_type(self, X):
        """Predict whether each sample ends up in a leaf node."""
        L = []
        for node_id in self.labels:
            L.append(self.get_node(node_id).is_leaf())
        return np.array(L)
    
    # Placeholder methods for sklearn compatibility

    def decision_path(self, X, check_input=True):
        pass
        
    def predict_proba(self, X, check_input=True):
        pass

    def predict_log_proba(self, X):
        pass

    def score(self, X, y=None):
        pass


    # Tree structure access methods
    
    def get_leaves(self, filter_func=None):
        """Get all leaf nodes, optionally filtered."""
        return list(
            filter(filter_func, self.incremental_tree.leaves())
        )
    
    def get_nodes(self, filter_func=None):
        """Get all nodes, optionally filtered."""
        return list(
            filter(filter_func, self.incremental_tree.all_nodes())
        )

    # Override BaseDecisionTree
    def get_n_leaves(self):
        """Get the number of leaves in the tree."""
        check_is_fitted(self)
        return len(self.incremental_tree.leaves())
    
    def get_depth(self):
        """Get the maximum depth of the tree."""
        check_is_fitted(self)
        return self.incremental_tree.depth() # get max level of the entire tree


    # EXPORT

    def to_networkx(self):
        """
        Export the tree structure to NetworkX format for visualization.
        
        Creates a directed graph where nodes represent tree nodes
        and edges represent parent-child relationships with split conditions.
        """
        G = ig.Graph(directed=True)
        nr_vertices = len(self.get_nodes())
        
        # Add root node with sample count information
        nodes_size = []
        nodes_deviance = []
        queue = [G.add_vertex(
            shape="record", 
            label="{root|"+f"N: {self.get_root().count}"+"}", 
            ident=self.get_root().identifier,
            size=float(self.get_root().count), 
            style='filled'
            )]
        i = 1

        while queue: # gather all nodes, leaves will not be added by the loop
            e = queue.pop(0)
            node = self.get_node(e['ident'])

            if node is not None:
                # Add left child if exists
                if node.left_child is not None:
                    # Create vertex for left child
                    l1 = G.add_vertex(
                        shape="record", 
                        label="{"+f"id: {node.left_child.label}"+f" | N: {node.left_child.count}"+"}", 
                        ident=node.left_child.identifier, 
                        style='filled', 
                        size=float(node.left_child.count)
                        )
                    
                    # Determine split condition format based on feature type (equality for categorical features,
                    # inequality for numerical features )
                    if node.attribute in list(map(lambda n: n[1], node._categorical_features)): 
                        mode = "="
                    else:
                        mode = "<="
                    
                    # Add edge with split condition
                    G.add_edges(
                        [(e, l1)], 
                        dict(label=f"{node.attribute} {mode} {str(node.threshold)[:6] if '.' in str(node.threshold) else node.threshold}"))
                    queue.append(l1)

                # Add right child if exists
                if node.right_child is not None:
                    # Create vertex for right child
                    r1 = G.add_vertex(
                        shape="record", 
                        label="{"+f"id: {node.right_child.label}"+f" | N: {node.right_child.count}"+"}", 
                        ident=node.right_child.identifier, 
                        style='filled', 
                        size=float(node.right_child.count)
                        )
                    
                    # if node.attribute in list(map(lambda n: n[1], node._categorical_features)): 
                    #     mode = "!=" 
                    # else:
                    #     mode = ">"

                    # Add edge
                    G.add_edges(
                        [(e, r1)],
                        dict(label="")) # Empty label for right branch
                    queue.append(r1)
            i += 1

        # Store layout as (xn, yn) into each node and (xe, ye) into each edge
        # layout = buchheim(self)
        
        # Convert graph to NetworkX format
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
        """Get the root node of the tree."""
        return self.incremental_tree.get_root()
    
    def get_node(self, identifier):
        """Get a specific node by its identifier."""
        return self.incremental_tree.get_node(identifier)
