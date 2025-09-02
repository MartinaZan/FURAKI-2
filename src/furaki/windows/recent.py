from .base import Window
import numpy as np
from scipy.linalg import pinv
from ..kernels.utils import pdfsum

class Recent(Window):
    """
    The difference with Standard is that Recent does not have a reset time and can attempt
    a split as soon as it is active and has seen more than 2 points (can_split property).
    """
    def __init__(self, num_tracker, cat_tracker, size, **kwargs) -> None:
        super().__init__(num_tracker, cat_tracker, size)

        # Resolution of the grid for kernel/density calculations (default 100)
        self.gridsize = kwargs.get("gridsize", 100)
        
        # Minimum number of points to attempt a split (default = window size)
        self.min_split = kwargs.get("min_split", self.size)

        # Reset time: number of points after which the window is refreshed (default = 0.1)
        self.reset_time = kwargs.get("reset_time", 0.1)

        # Flag indicating whether the window is active (collecting current statistics)
        self.activate = False
        
        # Counter for points seen
        self.counter = 0

        # Lists of two elements: [0] = mean (or covariance) of reference window,
        # [1] = mean of current window
        self.mean = [0, 0]
        self.cov = [0, 0]

    @property
    def can_split(self):
        return self.activate and self.n_points > 2
    
    def learn_one(self, x):
        """
        Update the window with a single new instance x.
        """
        # Increment total number of points seen in this window
        self.n_points += 1

        # If the window is not active yet, update the reference trackers with the new instance. Once
        # the number of points reaches the predefined size, activate the window, compute the reference
        # probability distribution by summing the kernels of the reference trackers, merge reference
        # trackers to get the mean and covariance for the left (historical) windowand reset the reference
        # trackers to start collecting new statistics.
        if not self.activate:

            self.update_ref_trackers(x)

            if self.n_points == self.size:
                # Activate window once enough points are collected
                self.activate = True

                # Compute reference probability distribution from reference trackers
                self.reference = pdfsum(self.get_ref_kernels())
                
                # Compute mean and covariance of the reference (left) window
                self.mean[0], self.cov[0] = self.merge(self.get_ref_trackers())
                
                # Reset counter for the next phase
                self.n_points = 0
        
        # If the window is already active, increment the counter used for periodic refresh, update
        # the current trackers with the new instance, merge the current trackers to compute the mean
        # and covariance of the current window, compute the current probability distribution (self.current)
        # by summing the kernels of the current trackers.
        else:
            # Window is active: update current statistics
            self.update_cur_trackers(x)

            # Merge current trackers to get mean and covariance for current window
            self.mean[1], self.cov[1] = self.merge(self.get_cur_trackers())
            
            # Compute current probability distribution from current trackers
            self.current = pdfsum(self.get_cur_kernels())



    def refresh(self, result):
        """
        Refresh the node's internal statistics only after a split.
        """
        # If a split occurred (result is True), it freezes left and right node statistics (mean
        # and pseudo-inverse of covariance), updates the reference distribution to the current
        # one, DUBBIO (self.mean[0] = self.mean[1] e self.cov[0] = self.cov[1]), resets the current
        # trackers to start collecting new statistics, and resets the counters.
        if result:
            self.frozen_left = (self.mean[0], pinv(self.cov[0]))
            self.frozen_right = (self.mean[1], pinv(self.cov[1]))
            
            self.reference = self.current
            self.mean[0] = self.mean[1]
            self.cov[0] = self.cov[1]
            
            self.reset_cur_trackers()
            self.n_points = 0
            self.counter = 0
            

    def merge(self, tracker_list):
        """
        Merge multiple trackers into a single mean vector and covariance matrix.
        It takes as input a list of trackers (numerical and/or categorical) and returns the combined
        mean vector and covariance matrix.

        NOTE: Instead of summing numerical and categorical covariances, we might consider using a
        block-diagonal covariance matrix to better preserve the structure of each feature type.
        """

        if len(tracker_list) > 1:
            # Concatenate the mean vectors of all trackers into a single mean vector
            mean = np.concatenate([list(tracker.getmean()) for tracker in tracker_list], axis=0)
            h = len(mean)

            # Combine the variances and covariances of all trackers into a single covariance matrix
            var = np.concatenate([tracker.getvar() for tracker in tracker_list], axis=0)
            cov = [tracker.getcov() for tracker in tracker_list]
            
            # Fill the new covariance matrix
            new_cov = np.zeros((h, h))
            for i in range(h):
                for j in range(h):
                    if i == j:  # Diagonal = variance
                        new_cov[i,j] = var[i]
                    else:       # Off-diagonal = sum of numerical and categorical covariances
                        new_cov[i, j] = cov[0][int(i % len(cov[0])), int(j % len(cov[0]))] + cov[1][int(i % len(cov[1])), int(j % len(cov[1]))]
            return mean, new_cov
        else:
            # If only one tracker, return its mean and covariance directly
            return tracker_list[0].getmean(), tracker_list[0].getcov()