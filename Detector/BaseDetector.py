import itertools
import numpy as np


def base_distance_enumeration(data_factory, hist_bin_num,
                              features, distance_thresholds,
                              diff_threshold,
                              prevalent_patterns):
    for dist in distance_thresholds:
        pi_distributions = data_factory.get_pi_distribution(hist_bin_num,
                                                            features,
                                                            dist)
        if np.linalg.norm(pi_distributions[0] - pi_distributions[1], 1) >= diff_threshold:
            prevalent_patterns.append((features, dist,
                                       np.linalg.norm(pi_distributions[0] - pi_distributions[1], 1)))
            print(prevalent_patterns[-1])
    return pi_distributions


def colocation_pattern_generator(labels, pattern_cardinality):
    for c in itertools.combinations(labels, pattern_cardinality):
        yield sorted(c)


class BaseDetector:
    def __init__(self,
                 data_factory,
                 diff_threshold, hist_bin_num, distance_thresholds):
        self.data_factory = data_factory
        self.labels = data_factory.labels

        self.diff_threshold = diff_threshold
        self.hist_bin_num = hist_bin_num
        self.distance_thresholds = distance_thresholds

        print('Data: {}; diff_threshold: {}; hist_bin_num: {}; distance_thresholds: {}'.format(
            self.data_factory, self.diff_threshold, self.hist_bin_num, self.distance_thresholds))
