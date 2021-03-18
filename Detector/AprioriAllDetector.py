from .BaseDetector import BaseDetector
from .AprioriIDetector import apriori_i_distance_enumeration
from .AprioriIIDetector import apriori_ii_pattern_enumeration
import time
import sys


class AprioriAllDetector(BaseDetector):
    def __init__(self,
                 data_factory,
                 diff_threshold, hist_bin_num, distance_thresholds):
        super().__init__(data_factory,
                         diff_threshold, hist_bin_num, distance_thresholds)
        print('AprioriAllDetector')

    def process(self):
        ans = []
        start_time = time.time()
        apriori_ii_pattern_enumeration(self.labels,
                                       self.data_factory, self.hist_bin_num, self.distance_thresholds,
                                       self.diff_threshold,
                                       apriori_i_distance_enumeration,
                                       ans)
        end_time = time.time()
        print('Execution time: {}s'.format(end_time - start_time))
        sys.stdout.flush()
        return ans
