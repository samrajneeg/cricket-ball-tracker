import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, mean, id, n_init=3, max_age=30):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # state [x, y, vx, vy]
        dt = 1.
        self.kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.kf.P *= 10.
        self.kf.R *= 1.
        self.kf.Q = np.eye(4) * 0.01
        self.kf.x[:2,0] = mean.reshape((2,))
        self.time_since_update = 0
        self.id = id
        self.hits = 1
        self.age = 0
        self.n_init = n_init
        self.max_age = max_age


    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1


    def update(self, measurement):
        self.kf.update(measurement.reshape((2,1)))
        self.hits += 1
        self.time_since_update = 0


    def get_state(self):
        return self.kf.x[:2,0]


class Tracker:
    def __init__(self, max_distance=60, max_age=30, n_init=1):
        self.tracks = []
        self._next_id = 1
        self.max_distance = max_distance
        self.max_age = max_age
        self.n_init = n_init


    def predict(self):
        for t in self.tracks:
            t.predict()


    def update(self, detections):
    # detections: list of [x,y]
        if len(self.tracks) == 0:
            for d in detections:
                tr = Track(np.array(d), self._next_id, n_init=self.n_init, max_age=self.max_age)
                self._next_id += 1
                self.tracks.append(tr)
            return


        # cost matrix
        M = np.zeros((len(self.tracks), len(detections)), dtype=float)
        for i, tr in enumerate(self.tracks):
            for j, d in enumerate(detections):
                diff = tr.get_state() - d
                M[i,j] = np.linalg.norm(diff)
        row_idx, col_idx = linear_sum_assignment(M)


        assigned_tracks = set()
        assigned_dets = set()
        # update assigned
        for r,c in zip(row_idx, col_idx):
            if M[r,c] > self.max_distance:
                continue
            self.tracks[r].update(np.array(detections[c]))
            assigned_tracks.add(r)
            assigned_dets.add(c)


        # unmatched detections -> new tracks
        for j, d in enumerate(detections):
            if j not in assigned_dets:
                tr = Track(np.array(d), self._next_id, n_init=self.n_init, max_age=self.max_age)
                self._next_id += 1

        
        # age and remove old tracks
        new_tracks = []
        for i, tr in enumerate(self.tracks):
            if i in assigned_tracks:
                new_tracks.append(tr)
            else:
                if tr.time_since_update < tr.max_age:
                    new_tracks.append(tr)
        self.tracks = new_tracks
    

    def get_active_tracks(self):
        # return list of tuples (id, x, y, is_confirmed)
        out = []
        for tr in self.tracks:
            state = tr.get_state()
            is_conf = tr.hits >= tr.n_init and tr.time_since_update == 0
            out.append((tr.id, float(state[0]), float(state[1]), int(is_conf)))
        return out