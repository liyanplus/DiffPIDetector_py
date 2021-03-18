from Detector import AprioriAllDetector, AprioriIIDetector, AprioriIDetector, BaselineDetector
from DataSource import UMichDataFactory
from PointSet import MultivariatePointSet
from collections import defaultdict
import numpy as np


def distance_experiment_helper(data_factory, detector, distance_num):
    MultivariatePointSet.pi_computing_times = defaultdict(int)
    detector = detector(
        data_factory, 1, 10, np.linspace(1, 200, distance_num).tolist()
    )
    detector.process()


for _ in range(3):
    df = UMichDataFactory(3)
    for dn in range(5, 21, 5):
        for d in [AprioriAllDetector, AprioriIIDetector, AprioriIDetector, BaselineDetector]:
            distance_experiment_helper(df, d, dn)
