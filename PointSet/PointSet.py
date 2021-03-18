from scipy.spatial import KDTree
from collections import defaultdict, deque
import numpy as np


class MultivariatePointSet:
    pi_computing_times = defaultdict(int)

    def __init__(self):
        self.points = defaultdict(list)

    def add_point(self, x, y, label):
        if label in self.points and type(self.points[label]) is not list:
            print('Cannot add point {} {} {}'.format(x, y, label))
        else:
            self.points[label].append([x, y])

    def build_index(self):
        ans = dict()
        for label, pts in self.points.items():
            if type(pts) is list:
                ans[label] = KDTree(np.array(pts))
            else:
                ans[label] = pts
        self.points = ans

    def _get_cliques(self, features, distance_threshold):
        if len(features) < 2:
            print('The input labels ({}) for cliques are invalid'.format(features))

        if type(self.points[features[0]]) is not KDTree:
            self.build_index()

        cache = deque([i] for i in range(self.points[features[0]].data.shape[0]))
        ans = []
        while len(cache):
            curr = cache.pop()
            if len(curr) == len(features):
                ans.append(curr)
            else:
                next_label = features[len(curr)]
                next_neighbors = set(range(self.points[next_label].data.shape[0]))
                for label, pt_idx in zip(features[:len(curr)], curr):
                    next_neighbors.intersection_update(
                        set(self.points[next_label].query_ball_point(
                            self.points[label].data[pt_idx],
                            distance_threshold)))
                    if len(next_neighbors) == 0:
                        break
                cache.extend(list(curr) + [nn] for nn in next_neighbors)
        return ans

    def get_participation_index(self, features, distance_threshold):
        if len(features) == 0:
            return 0

        MultivariatePointSet.pi_computing_times[len(features)] += 1

        for label in features:
            if label not in self.points or self.points[label].data.shape[0] == 0:
                return 0

        if type(self.points[features[0]]) is not KDTree:
            self.build_index()

        cliques = self._get_cliques(features, distance_threshold)
        return min(
            len(set(c[l_idx] for c in cliques)) / self.points[label].data.shape[0]
            for l_idx, label in enumerate(features)
        )
