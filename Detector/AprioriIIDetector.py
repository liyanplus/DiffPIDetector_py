import numpy as np
from .BaseDetector import BaseDetector, colocation_pattern_generator, base_distance_enumeration
from .AprioriIDetector import get_cumpi_distributions, apriori_bounds
import sys
from PointSet import MultivariatePointSet
import time


def apriori_ii_pattern_enumeration(labels,
                                   data_factory, hist_bin_num, distance_thresholds,
                                   diff_threshold,
                                   distance_enumeration_method,
                                   prevalent_patterns):
    def check_subpatterns_cumpr(curr_features):
        lower_cumpi = np.zeros((2, hist_bin_num))
        for idx in range(len(curr_features)):
            subset_feature_tuple = tuple(sorted(curr_features[:idx] + curr_features[idx + 1:]))
            if subset_feature_tuple in cache:
                lower_cumpi = np.max(np.stack((lower_cumpi, cache[subset_feature_tuple]),
                                              axis=2), axis=2)
            else:
                return np.ones((2, hist_bin_num))
        return lower_cumpi

    cache = dict()
    for pattern_cardinality in range(2, len(labels) + 1):
        print('Pattern cardinality: {}'.format(pattern_cardinality))
        updated_cache = dict()
        for features in colocation_pattern_generator(labels, pattern_cardinality):
            if len(features) > 2:
                prev_lower_cumpi = check_subpatterns_cumpr(features)
                max_prev_diff, _ = apriori_bounds(
                    prev_lower_cumpi, hist_bin_num
                )
                if max_prev_diff < diff_threshold:
                    break

            last_pis = distance_enumeration_method(data_factory, hist_bin_num,
                                                   features, distance_thresholds,
                                                   diff_threshold, prevalent_patterns)

            updated_cache[tuple(sorted(features))] = get_cumpi_distributions(last_pis)
            sys.stdout.flush()
        print('PI computing times: {}'.format(MultivariatePointSet.pi_computing_times[pattern_cardinality]))
        cache = updated_cache


class AprioriIIDetector(BaseDetector):
    def __init__(self,
                 data_factory,
                 diff_threshold, hist_bin_num, distance_thresholds):
        super().__init__(data_factory,
                         diff_threshold, hist_bin_num, distance_thresholds)
        print('AprioriIIDetector')

    def process(self):
        ans = []
        start_time = time.time()
        apriori_ii_pattern_enumeration(self.labels,
                                       self.data_factory, self.hist_bin_num, self.distance_thresholds,
                                       self.diff_threshold,
                                       base_distance_enumeration,
                                       ans)
        end_time = time.time()
        print('Execution time: {}s'.format(end_time - start_time))
        sys.stdout.flush()
        return ans
