from .base import Window
import numpy as np
from scipy.linalg import pinv
from ..kernels.utils import pdfsum

class Recent(Window):
    def __init__(self, num_tracker, cat_tracker, size, **kwargs) -> None:
        super().__init__(num_tracker, cat_tracker, size)
        self.gridsize = kwargs.get("gridsize", 100)
        self.min_split = kwargs.get("min_split", self.size)
        self.reset_time = kwargs.get("reset_time", 0.1)
        self.activate = False
        self.counter = 0
        self.mean = [0, 0]
        self.cov = [0, 0]

    @property
    def can_split(self):
        return self.activate and self.n_points > 2
    
    def learn_one(self, x):
        self.n_points += 1

        if not self.activate:

            self.update_ref_trackers(x)

            if self.n_points == self.size:
                self.activate = True
                self.reference = pdfsum(self.get_ref_kernels())
                self.mean[0], self.cov[0] = self.merge(self.get_ref_trackers())
                #self.reset_ref_trackers()
                self.n_points = 0
        
        else:
            self.update_cur_trackers(x)
            self.mean[1], self.cov[1] = self.merge(self.get_cur_trackers())
            # self.update_ref_trackers(x)
            self.current = pdfsum(self.get_cur_kernels())



    def refresh(self, result):
        if result:
            #self.num_tracker_ref = copy.deepcopy(self.num_tracker_cur)
            #self.cat_tracker_ref = copy.deepcopy(self.cat_tracker_cur)
            self.frozen_left = (self.mean[0], pinv(self.cov[0]))
            self.frozen_right = (self.mean[1], pinv(self.cov[1]))
            
            self.reference = self.current
            self.mean[0] = self.mean[1]
            self.cov[0] = self.cov[1]
            # self.current = None
            # self.reference = pdfsum(self.get_ref_kernels(), self.gridsize)
            # self.current = pdfsum(self.get_cur_kernels(), self.gridsize)
            # self.swap_trackers()
            # self.reset_ref_trackers()
            self.reset_cur_trackers()
            self.n_points = 0
            self.counter =0
            

    def merge(self, tracker_list):
        if len(tracker_list) > 1:
            mean = np.concatenate([list(tracker.getmean()) for tracker in tracker_list], axis=0)
            h = len(mean)
            var = np.concatenate([tracker.getvar() for tracker in tracker_list], axis=0)
            cov = [tracker.getcov() for tracker in tracker_list]
            new_cov = np.zeros((h, h))
            for i in range(h):
                for j in range(h):
                    if i == j:
                        new_cov[i,j] = var[i]
                    else:
                        new_cov[i, j] = cov[0][int(i%len(cov[0])), int(j%len(cov[0]))] + cov[1][int(i%len(cov[1])), int(j%len(cov[1]))]
            return mean, new_cov
        else:
            return tracker_list[0].getmean(), tracker_list[0].getcov()