import numpy as np
from .BaseDetector import BaseDetector, colocation_pattern_generator
import bisect
from collections import defaultdict
import time
from PointSet import MultivariatePointSet
import sys


def get_cumpi_distributions(pi_distributions):
    return np.cumsum(pi_distributions, axis=1)


def apriori_bounds(cumpi_lower_bound, hist_bin_num,
                   cumpi_upper_bound=None):
    if cumpi_upper_bound is None:
        cumpi_upper_bound = np.ones((2, hist_bin_num))

    if np.sum(cumpi_upper_bound - cumpi_lower_bound < -1e-3):
        print('Cumulative pi is invalid.\n{}; {}.'.format(cumpi_upper_bound, cumpi_lower_bound))

    class1_cumpi = np.vstack((cumpi_lower_bound[0], cumpi_upper_bound[0]))
    class2_cumpi = np.vstack((cumpi_lower_bound[1], cumpi_upper_bound[1]))

    def float_unique(raw_array):
        TOL = 1.0e-3
        tmp = raw_array.flatten()
        tmp.sort()
        tmp2 = np.append(True, np.diff(tmp))
        return tmp[tmp2 > TOL]

    class1_breakpoints = float_unique(class1_cumpi)
    class2_breakpoints = float_unique(class2_cumpi)

    def get_breakpoints_index_range(breakpoints, lo_val, hi_val):
        lo_idx = bisect.bisect_left(breakpoints, lo_val)
        hi_idx = bisect.bisect_left(breakpoints, hi_val)
        return lo_idx if lo_idx != len(breakpoints) else len(breakpoints) - 1, \
               hi_idx if hi_idx != len(breakpoints) else len(breakpoints) - 1

    def update_state(prev1, prev2, curr1, curr2,
                     prev_idx1, prev_idx2,
                     curr_idx1, curr_idx2):
        if prev1 > curr1 + 1e-4 or prev2 > curr2 + 1e-4:
            return
        cur_diff = abs((curr1 - prev1) -
                       (curr2 - prev2))

        if curr_idx2 not in curr_max[curr_idx1]:
            curr_max[curr_idx1][curr_idx2] = 0
            curr_min[curr_idx1][curr_idx2] = float('inf')

        curr_max[curr_idx1][curr_idx2] = max(
            curr_max[curr_idx1][curr_idx2],
            prev_max[prev_idx1][prev_idx2] + cur_diff
        )
        curr_min[curr_idx1][curr_idx2] = min(
            curr_min[curr_idx1][curr_idx2],
            prev_min[prev_idx1][prev_idx2] + cur_diff
        )

    prev_max = {-1: {-1: 0}}
    prev_min = {-1: {-1: 0}}
    for idx in range(hist_bin_num):
        curr_max = defaultdict(dict)
        curr_min = defaultdict(dict)

        class1_breakpoint_index_range = get_breakpoints_index_range(class1_breakpoints,
                                                                    class1_cumpi[0, idx],
                                                                    class1_cumpi[1, idx])
        class2_breakpoint_index_range = get_breakpoints_index_range(class2_breakpoints,
                                                                    class2_cumpi[0, idx],
                                                                    class2_cumpi[1, idx])
        for curr_class1_breakpoint_idx in range(class1_breakpoint_index_range[0], class1_breakpoint_index_range[1] + 1):
            curr_class1_cumpi = class1_breakpoints[curr_class1_breakpoint_idx]
            for curr_class2_breakpoint_idx in range(class2_breakpoint_index_range[0],
                                                    class2_breakpoint_index_range[1] + 1):
                curr_class2_cumpi = class2_breakpoints[curr_class2_breakpoint_idx]
                if idx == 0:
                    update_state(0, 0,
                                 curr_class1_cumpi, curr_class2_cumpi,
                                 -1, -1,
                                 curr_class1_breakpoint_idx, curr_class2_breakpoint_idx)
                else:
                    for prev_class1_breapoint_idx in prev_max.keys():
                        prev_class1_cumpi = class1_breakpoints[prev_class1_breapoint_idx]
                        for prev_class2_breapoint_idx in prev_max[prev_class1_breapoint_idx].keys():
                            prev_class2_cumpi = class2_breakpoints[prev_class2_breapoint_idx]
                            update_state(prev_class1_cumpi, prev_class2_cumpi,
                                         curr_class1_cumpi, curr_class2_cumpi,
                                         prev_class1_breapoint_idx, prev_class2_breapoint_idx,
                                         curr_class1_breakpoint_idx, curr_class2_breakpoint_idx)
        prev_max = curr_max
        prev_min = curr_min

    ans_max = 0
    ans_min = float('inf')
    for i1 in prev_max.keys():
        for i2 in prev_max[i1].keys():
            ans_max = max(ans_max, prev_max[i1][i2])
            ans_min = min(ans_min, prev_min[i1][i2])
    return ans_max, ans_min


def apriori_i_distance_enumeration(data_factory, hist_bin_num,
                                   features, distance_thresholds,
                                   diff_threshold,
                                   prevalent_patterns):
    def helper(lo_distance_idx, hi_distance_idx, lo_distance_pis, hi_distance_pis):
        if lo_distance_idx >= hi_distance_idx:
            return
        elif lo_distance_idx + 1 == hi_distance_idx:
            pi_diff_lo_distance = np.linalg.norm(lo_distance_pis[0] - lo_distance_pis[1], 1)
            if pi_diff_lo_distance >= diff_threshold:
                prevalent_patterns.append((features,
                                           distance_thresholds[lo_distance_idx],
                                           pi_diff_lo_distance))
                print(prevalent_patterns[-1])
        else:
            lo_distance_cumpis = get_cumpi_distributions(lo_distance_pis)
            hi_distance_cumpis = get_cumpi_distributions(hi_distance_pis)
            max_diff, min_diff = apriori_bounds(
                hi_distance_cumpis, hist_bin_num, lo_distance_cumpis
            )
            if max_diff < diff_threshold:
                return
            if min_diff >= diff_threshold:
                for dist in distance_thresholds[lo_distance_idx:hi_distance_idx]:
                    prevalent_patterns.append((features, dist, min_diff))
                    print(prevalent_patterns[-1])
                return
            mid_distance_idx = (lo_distance_idx + hi_distance_idx) // 2
            mid_distance_pis = data_factory.get_pi_distribution(hist_bin_num,
                                                                features,
                                                                distance_thresholds[mid_distance_idx])
            helper(lo_distance_idx, mid_distance_idx, lo_distance_pis, mid_distance_pis)
            helper(mid_distance_idx, hi_distance_idx, mid_distance_pis, hi_distance_pis)

    pis_1st = data_factory.get_pi_distribution(hist_bin_num, features, distance_thresholds[0])
    pis_last = data_factory.get_pi_distribution(hist_bin_num, features, distance_thresholds[-1])
    pi_diff_last = np.linalg.norm(pis_last[0] - pis_last[1], 1)
    if pi_diff_last >= diff_threshold:
        prevalent_patterns.append((features, distance_thresholds[-1], pi_diff_last))
        print(prevalent_patterns[-1])
    helper(0, len(distance_thresholds) - 1, pis_1st, pis_last)
    return pis_last


class AprioriIDetector(BaseDetector):
    def __init__(self,
                 data_factory,
                 diff_threshold, hist_bin_num, distance_thresholds):
        super().__init__(data_factory,
                         diff_threshold, hist_bin_num, distance_thresholds)
        print('AprioriIDetector')

    def process(self):
        ans = []
        start_time = time.time()
        for pattern_cardinality in range(2, len(self.labels) + 1):
            print('Pattern cardinality: {}'.format(pattern_cardinality))
            for features in colocation_pattern_generator(self.labels, pattern_cardinality):
                apriori_i_distance_enumeration(self.data_factory, self.hist_bin_num,
                                               features, self.distance_thresholds,
                                               self.diff_threshold, ans)
            print('PI computing times: {}'.format(MultivariatePointSet.pi_computing_times[pattern_cardinality]))
        end_time = time.time()
        print('Execution time: {}s'.format(end_time - start_time))
        sys.stdout.flush()
        return ans
