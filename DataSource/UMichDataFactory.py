import numpy as np
import random
from . import UMichDataSource
import bisect


class UMichDataFactory:
    _labels = 'Epithelial,Treg,APC,HelperT,CD4,CTLs,PDL1_CD3,PDL1_CD8,PDL1_FoxP3'.split(',')
    # labels = 'PDL1_FoxP3,PDL1_CD3,CD4'.split(',')
    data_dirs = [r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/Data/Anon_Group1/',
                 r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/Data/Anon_Group2/']

    def __init__(self,
                 label_num=None,
                 sample_size=None,
                 data_augmentation=None):
        self.sample_size = sample_size
        self.data_augmentation = data_augmentation

        if label_num is None:
            self.labels = UMichDataFactory._labels
        elif type(label_num) is int:
            if label_num < len(UMichDataFactory._labels):
                self.labels = random.sample(UMichDataFactory._labels, label_num)
            elif label_num == len(UMichDataFactory._labels):
                self.labels = UMichDataFactory._labels
                random.shuffle(self.labels)
            elif label_num > len(UMichDataFactory._labels):
                self.labels = UMichDataFactory._labels
                random.shuffle(UMichDataFactory._labels)
                print('Invalid input number of labels: {}'.format(label_num))
        elif type(label_num) is list:
            self.labels = label_num

    def __repr__(self):
        return 'UMich; Labels: {}'.format(self.labels) + \
               (' Sample size: {};'.format(self.sample_size) if self.sample_size is not None else '') + \
               (' Data augmentation: {}'.format(self.data_augmentation) if self.data_augmentation is not None else '')

    def __str__(self):
        return 'UMich; Labels: {}'.format(self.labels) + \
               (' Sample size: {};'.format(self.sample_size) if self.sample_size is not None else '') + \
               (' Data augmentation: {}'.format(self.data_augmentation) if self.data_augmentation is not None else '')

    def _get_augmented_data(self, class_idx):
        filepaths = list(UMichDataSource.file_system_scrawl(UMichDataFactory.data_dirs[class_idx], '.txt'))
        if self.data_augmentation is None:
            return filepaths
        if type(self.data_augmentation) is int and self.data_augmentation < len(filepaths):
            return random.sample(filepaths, self.data_augmentation)
        elif type(self.data_augmentation) is int and self.data_augmentation == len(filepaths):
            random.shuffle(filepaths)
            return filepaths
        elif type(self.data_augmentation) is int and self.data_augmentation > len(filepaths):
            return random.choices(filepaths, k=self.data_augmentation)

    def get_pi_distribution(self, hist_bin_num, features, distance_threshold):
        hists = np.zeros((2, hist_bin_num))

        def pi_helper(class_idx):
            def get_sample_size():
                if self.sample_size is None:
                    return float('inf')
                if self.sample_size != 'freeman':
                    return self.sample_size
                if len(pi_cache) <= 10 * hist_bin_num:
                    return float('inf')
                return np.power(
                    hist_bin_num * 2 * abs(pi_cache[int(len(pi_cache) * 0.1)] - pi_cache[int(len(pi_cache) * 0.9)]), 3)

            if self.sample_size == 'freeman':
                pi_cache = []

            for path in self._get_augmented_data(class_idx):
                ps = UMichDataSource.read_file(path)
                ps.build_index()
                pi = ps.get_participation_index(features, distance_threshold)
                if pi == 1:
                    hists[class_idx, hist_bin_num - 1] += 1
                else:
                    hists[class_idx, int(pi * hist_bin_num)] += 1

                if self.sample_size == 'freeman':
                    bisect.insort_left(pi_cache, pi)

                if np.sum(hists[class_idx]) > get_sample_size():
                    break
            hists[class_idx] = hists[class_idx] / np.sum(hists[class_idx])

        pi_helper(0)
        pi_helper(1)
        return hists
