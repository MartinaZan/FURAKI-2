from __future__ import annotations
from abc import ABCMeta, abstractmethod

from treelib import Tree
from treelib.exceptions import *

import copy
import pandas as pd

from .node import IncrementalTreeNode


class IncrementalTree(Tree):
    """
    An incremental decision tree that extends treelib.Tree.
    
    This tree can learn from streaming data by continuously updating its structure
    as new instances arrive, without requiring a complete rebuild.
    """

    def __init__(
        self,
        *,
        criterion = "infogain",             # 
        splitter = "g-test",                # Statistical test for determining splits
        window = 'standard',                # Type of sliding window (recent/standard)
        window_params = {},                 # Parameters for the sliding window
        max_depth = None,                   # Maximum depth of the tree
        min_samples_split = 2,              # Minimum samples required to split a node
        min_samples_leaf = 1,               # Minimum samples required in a leaf node
        min_weight_fraction_leaf = 0.0,     # Minimum fraction of samples in a leaf
        max_features = None,                # Maximum features to consider for splits
        random_state = None,                # Random state for reproducibility
        max_leaf_nodes = None,              # Maximum number of leaf nodes
        min_impurity_decrease = 0.0,        # Minimum impurity decrease required for split
        numerical_tracker = "covariance",   # Method to track numerical feature statistics
        numerical_tracker_params = {},      # Parameters for numerical tracker
        categorical_tracker = "frequency",  # Method to track categorical feature statistics
        categorical_tracker_params = {},    # Parameters for categorical tracker
        threshold_criterion = "midpoint",   # Criterion for determining split thresholds
        baseline_wait = 1000,               # Number of samples to wait before baseline establishment
        sample_test_size = 1000,            # Sample size for statistical tests
        alpha = 0.95,                       # Confidence level for statistical test
        **kwargs
    ):
        super().__init__(
            node_class=IncrementalTreeNode
        )
        
        # dtreeviz
        self.criterion = criterion

        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

        self._categorical_tracker = categorical_tracker
        self._numerical_tracker = numerical_tracker
        self._numerical_tracker_params = numerical_tracker_params
        self._categorical_tracker_params = categorical_tracker_params
        self._splitter = splitter
        self._criterion = criterion
        self._window = window
        self._window_params = window_params
        self._threshold_criterion = threshold_criterion
        self._baseline_wait=baseline_wait
        self._sample_test_size = sample_test_size
        self._confidence = alpha

        self.num_splits = 0             # Counter for the number of splits performed
        self.split_dict = {0: 'root'}   # Dictionary to track split labels

        # Sets to track nodes that are candidates for pruning or merging
        self._prune_candidates = set()
        self._merge_candidates = set()

        # Track when the last split occurred
        self.split_time = -1


    """Properties"""
    @property
    def count(self) -> int:
        """Calculate the total number of instances processed by the tree."""
        s = 0
        for node in self.all_nodes():
            s += node.count     # Number of instances seen by each node
        return s

    @property
    def n_features(self) -> int:
        """Get the total number of features (numerical + categorical)."""
        return len(self._numerical_features) + len(self._categorical_features)
    
    

    """Public methods"""

    # Override BaseDecisionTree
    def get_root(self) -> IncrementalTreeNode:
        """Get the root node of the tree."""
        if hasattr(self, "root"):
            return self.root
        return None
    
    def get_leaves(self, filter_func=None):
        """Get all leaf nodes in the tree, optionally filtered."""
        return list(
            filter(filter_func, self.leaves())
        )
    
    def get_nodes(self, filter_func=None) -> list[IncrementalTreeNode]:
        """Get all nodes in the tree, optionally filtered."""
        return list(
            filter(filter_func, self.all_nodes())
        )
   

    """Private methods"""


    def _learn_one(self, X):
        """
        This function updates the decision tree with a single instance X.

        This is the core incremental learning method that:
          1. Creates root node if tree is empty
          2. Routes instance to appropriate leaf node
          3. Updates the leaf node with the instance
          4. Handles node splitting when criteria are met
        """
        # If the tree is empty (i.e., there is no root node), create the root node with initial
        # data and set its label (0)
        if self.get_root() is None:
            self.root = self.create_node(
                data=self.set_data()
            )
            self.root.label = 0
        
        # Route the instance through the tree to find the appropriate leaf node
        found_node = self.root.filter_instance_to_leaf(X)

        # If such a node is found, update it with the instance X. Then, check if the node should be split.
        if found_node is not None:
            # Update the found node with the new instance
            found_node._learn_one(X)

            # If a split occurs, create two new child nodes for the split, copying the window from the
            # parent node. Assign a new label to one of the child nodes. Finally, return the identifier of
            # the node where the instance X was processed.
            if found_node.split(X):
                self.num_splits += 1            # Increment the split counter
                self.split_time = self.count    # Record the split time
                
                # Create the left child node: copy parent's window and inherit parent's label
                n1 = self.create_node(
                    parent=found_node.identifier,
                    data=self.set_data()
                )
                n1._window = copy.deepcopy(found_node._window)
                n1.label = found_node.label

                # Create the right child node: copy parent's window and assign a new label
                n2 = self.create_node(
                    parent=found_node.identifier,
                    data=self.set_data()
                )
                n2._window = copy.deepcopy(found_node._window)
                n2.label = self._get_next_label()

        # Return the identifier of the leaf node where the instance was added
        return found_node.identifier


    """Public methods"""


    """Private methods"""

    def set_data(self):
        """Create a configuration dictionary for tree nodes."""
        return dict(
            tree = self,
            max_depth = self.max_depth,
            alpha = self._confidence,
            numerical_features = self._numerical_features,
            categorical_features = self._categorical_features,
            min_samples_split = self.min_samples_split,
            sample_test_size = self._sample_test_size,
            numerical = self._numerical_tracker,
            numerical_params = self._numerical_tracker_params,
            categorical = self._categorical_tracker,
            categorical_params = self._categorical_tracker_params,
            splitter = self._splitter,
            criterion = self._criterion,
            window = self._window,
            window_params = self._window_params,
            threshold_criterion = self._threshold_criterion
        )
    
    def _extract_features(self, X:pd.DataFrame):
        """Analyze DataFrame to identify numerical and categorical features."""
        self._numerical_features = []
        self._categorical_features = []
        for i, dt in enumerate(X.dtypes):
            if "float" in str(dt) or "int" in str(dt):
                self._numerical_features.append((i, X.columns[i]))
            if "obj" in str(dt) or "cat" in str(dt): # value dt is an object, possibly string
                self._categorical_features.append((i, X.columns[i]))

    def _get_next_label(self):
        """
        Generate the next unique label for a new node.
        
        Labels are used to identify different branches/regions in the tree.
        This method ensures each new split gets a unique identifier."""

        # Find the maximum existing label and increment by 1
        n = max(list(self.split_dict.keys())) + 1

        # Initialize the new label in the split dictionary
        self.split_dict[n] = 0
        
        return n

