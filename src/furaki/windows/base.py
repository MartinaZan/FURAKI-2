from abc import abstractmethod
from ..kernels import Grid

class Window():
    def __init__(self, num_tracker, cat_tracker, size) -> None:
        self.size = size
        self.n_points = 0
        self.counter = 0
        self.activate = False
        self.reference = None
        self.current = None
        self.frozen_left = None
        self.frozen_right = None
        self.num_class =num_tracker
        self.cat_class=cat_tracker
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
        z = []
        if self.num_tracker_cur is not None:
            z.append(self.num_tracker_cur.kernel)
        if self.cat_tracker_cur is not None:
            z.append(self.cat_tracker_cur.kernel)
        return z
    
    def get_cur_trackers(self):
        z = []
        if self.num_tracker_cur is not None:
            z.append(self.num_tracker_cur)
        if self.cat_tracker_ref is not None:
            z.append(self.cat_tracker_cur)
        return z
    
    def get_ref_kernels(self):
        z = []
        if self.num_tracker_ref is not None:
            z.append(self.num_tracker_ref.kernel)
        if self.cat_tracker_ref is not None:
            z.append(self.cat_tracker_ref.kernel)
        return z
    
    def get_ref_trackers(self):
        z = []
        if self.num_tracker_ref is not None:
            z.append(self.num_tracker_ref)
        if self.cat_tracker_ref is not None:
            z.append(self.cat_tracker_ref)
        return z
    
    def get_reference(self):
        p = self.reference.get(normalize=True, as_prod=False)
        return p
    
    def get_current(self):
        p = self.current.get(normalize=True, as_prod=False)
        return p
    
    def update_ref_trackers(self, x):
        if self.num_tracker_ref is not None:
            self.num_tracker_ref.update(x)
        if self.cat_tracker_ref is not None:
            self.cat_tracker_ref.update(x)

    def update_cur_trackers(self, x):
        if self.num_tracker_cur is not None:
            self.num_tracker_cur.update(x)
        if self.cat_tracker_cur is not None:
            self.cat_tracker_cur.update(x)
    
    def reset_cur_trackers(self):
        if self.num_tracker_cur is not None:
            self.num_tracker_cur.reset()
        if self.cat_tracker_cur is not None:
            self.cat_tracker_cur.reset()

    def reset_ref_trackers(self):
        if self.num_tracker_ref is not None:
            self.num_tracker_ref.reset()
        if self.cat_tracker_ref is not None:
            self.cat_tracker_ref.reset()

    def get_threshold_by_feature_index(self, feature_index):
        for stats_tracker in self.get_cur_trackers():
            if feature_index in stats_tracker.feature_indices:
                break
            # i += stats_tracker.n_features
        
        return stats_tracker.get_threshold(feature_index)