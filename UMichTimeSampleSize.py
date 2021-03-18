from Detector import AprioriAllDetector, AprioriIIDetector, AprioriIDetector, BaselineDetector
from DataSource import UMichDataFactory
from PointSet import MultivariatePointSet
from collections import defaultdict
import random


def sample_size_experiment_helper(data_factory, detector):
    MultivariatePointSet.pi_computing_times = defaultdict(int)
    detector = detector(
        data_factory, 1, 10, [1, 50, 100, 150, 200]
    )
    detector.process()


for ag in range(200, 351, 50):
    for _ in range(3):
        labels = random.sample(UMichDataFactory._labels, 3)
        freeman_df = UMichDataFactory(labels, 'freeman', ag)
        base_df = UMichDataFactory(labels, None, ag)

        sample_size_experiment_helper(freeman_df, AprioriAllDetector)
        sample_size_experiment_helper(base_df, AprioriAllDetector)
