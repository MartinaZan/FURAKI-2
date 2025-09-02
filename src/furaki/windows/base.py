from abc import abstractmethod
from ..kernels import Grid

class Window():
    """
    Base class for a data window with numerical and categorical variables.
    Manages separate trackers for numerical and categorical data, both for the
    reference (historical) window and the current window.
    """
    def __init__(self, num_tracker, cat_tracker, size) -> None:
        self.size = size                # Number of points needed to activate the window
        self.n_points = 0               # Counter for total points seen so far
        self.counter = 0                # Counter for periodic updates (used in Standard window)
        self.activate = False           # Flag: is the window active? True after collecting 'size' points
        self.reference = None           # Reference probability distribution
        self.current = None             # Current probability distribution
        self.frozen_left = None         # Frozen statistics for left node after a split
        self.frozen_right = None        # Frozen statistics for right node after a split
        self.num_class = num_tracker    # Tracker for numerical features
        self.cat_class = cat_tracker    # Tracker for categorical features

        # Initialize trackers for reference and current windows
        self.num_tracker_ref = num_tracker() if num_tracker is not None else None
        self.cat_tracker_ref = cat_tracker() if cat_tracker is not None else None
        self.num_tracker_cur = num_tracker() if num_tracker is not None else None
        self.cat_tracker_cur = cat_tracker() if cat_tracker is not None else None

    @abstractmethod
    def learn_one(self, x):
        pass

    @abstractmethod
    def refresh(self, result):
        pass

    @abstractmethod
    def can_split(self):
        pass

    def get_cur_kernels(self):
        """Return kernels (numerical/categorical) from current trackers."""
        z = []
        if self.num_tracker_cur is not None:
            z.append(self.num_tracker_cur.kernel)
        if self.cat_tracker_cur is not None:
            z.append(self.cat_tracker_cur.kernel)
        return z
    
    def get_cur_trackers(self):
        """Return current trackers for numerical and categorical features."""
        z = []
        if self.num_tracker_cur is not None:
            z.append(self.num_tracker_cur)
        if self.cat_tracker_ref is not None:
            z.append(self.cat_tracker_cur)
        return z
    
    def get_ref_kernels(self):
        """Return kernels (numerical/categorical) from reference trackers."""
        z = []
        if self.num_tracker_ref is not None:
            z.append(self.num_tracker_ref.kernel)
        if self.cat_tracker_ref is not None:
            z.append(self.cat_tracker_ref.kernel)
        return z
    
    def get_ref_trackers(self):
        """Return reference trackers for numerical and categorical features."""
        z = []
        if self.num_tracker_ref is not None:
            z.append(self.num_tracker_ref)
        if self.cat_tracker_ref is not None:
            z.append(self.cat_tracker_ref)
        return z
    
    def get_reference(self):
        """
        Return normalized probability distribution from reference trackers.
        (with as_prod=False we get the per-feature likelihoods (one value per feature), without
        multiplying them)
        """
        p = self.reference.get(normalize=True, as_prod=False)
        return p
    
    def get_current(self):
        """
        Return normalized probability distribution from current trackers.
        (with as_prod=False we get the per-feature likelihoods (one value per feature), without
        multiplying them)
        """
        p = self.current.get(normalize=True, as_prod=False)
        return p
    
    def update_ref_trackers(self, x):
        """Update reference trackers with a new sample x."""
        if self.num_tracker_ref is not None:
            self.num_tracker_ref.update(x)
        if self.cat_tracker_ref is not None:
            self.cat_tracker_ref.update(x)

    def update_cur_trackers(self, x):
        """Update current trackers with a new sample x."""
        if self.num_tracker_cur is not None:
            self.num_tracker_cur.update(x)
        if self.cat_tracker_cur is not None:
            self.cat_tracker_cur.update(x)
    
    def reset_cur_trackers(self):
        """Reset current trackers (start a new current window)."""
        if self.num_tracker_cur is not None:
            self.num_tracker_cur.reset()
        if self.cat_tracker_cur is not None:
            self.cat_tracker_cur.reset()

    def reset_ref_trackers(self):
        """Reset reference trackers (start a new baseline window)."""
        if self.num_tracker_ref is not None:
            self.num_tracker_ref.reset()
        if self.cat_tracker_ref is not None:
            self.cat_tracker_ref.reset()

    def get_threshold_by_feature_index(self, feature_index):
        """
        Return the threshold value for a specific feature index from the current trackers.
        Used when deciding where to split.

        TO CLARIFY: when is this threshold used?
        """
        for stats_tracker in self.get_cur_trackers():
            if feature_index in stats_tracker.feature_indices:
                break
            # i += stats_tracker.n_features
        
        return stats_tracker.get_threshold(feature_index)