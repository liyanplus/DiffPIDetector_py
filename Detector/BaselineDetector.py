from .BaseDetector import BaseDetector, colocation_pattern_generator, base_distance_enumeration
import time
from PointSet import MultivariatePointSet
import sys


class BaselineDetector(BaseDetector):
    def __init__(self,
                 data_factory,
                 diff_threshold, hist_bin_num, distance_thresholds):
        super().__init__(data_factory,
                         diff_threshold, hist_bin_num, distance_thresholds)
        print('BaselineDetector')

    def process(self):
        ans = []
        start_time = time.time()
        for pattern_cardinality in range(2, len(self.labels) + 1):
            print('Pattern cardinality: {}'.format(pattern_cardinality))
            for features in colocation_pattern_generator(self.labels, pattern_cardinality):
                base_distance_enumeration(self.data_factory, self.hist_bin_num,
                                          features, self.distance_thresholds,
                                          self.diff_threshold, ans)
            print('PI computing times: {}'.format(MultivariatePointSet.pi_computing_times[pattern_cardinality]))
        end_time = time.time()
        print('Execution time: {}s'.format(end_time - start_time))
        sys.stdout.flush()
        return ans
