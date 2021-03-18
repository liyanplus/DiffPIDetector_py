from Detector import AprioriAllDetector, AprioriIIDetector, AprioriIDetector, BaselineDetector
from DataSource import UMichDataFactory
from PointSet import MultivariatePointSet
from collections import defaultdict


def feature_num_experiment_helper(data_factory, detector):
    MultivariatePointSet.pi_computing_times = defaultdict(int)
    detector = detector(
        data_factory, 1, 10, [1, 50, 100, 150, 200]
    )
    detector.process()


for fn in range(3, 6):
    for _ in range(3):
        df = UMichDataFactory(fn)
        for d in [AprioriAllDetector, AprioriIIDetector, AprioriIDetector, BaselineDetector]:
            feature_num_experiment_helper(df, d)
