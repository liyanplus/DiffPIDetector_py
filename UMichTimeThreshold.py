from Detector import AprioriAllDetector, AprioriIIDetector, AprioriIDetector, BaselineDetector
from DataSource import UMichDataFactory
from PointSet import MultivariatePointSet
from collections import defaultdict
import numpy as np


def distance_experiment_helper(data_factory, detector, diff_threshold):
    MultivariatePointSet.pi_computing_times = defaultdict(int)
    detector = detector(
        data_factory, diff_threshold, 10, np.linspace(1, 200, 10).tolist()
    )
    detector.process()


for _ in range(3):
    df = UMichDataFactory(5)
    for dt in [0.4, 0.7, 1, 1.3, 1.6]:
        for d in [AprioriAllDetector, AprioriIIDetector, AprioriIDetector, BaselineDetector]:
            distance_experiment_helper(df, d, dt)
