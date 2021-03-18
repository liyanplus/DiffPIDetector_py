from Detector import AprioriAllDetector
from DataSource import UMichDataFactory

detector = AprioriAllDetector(
    UMichDataFactory(['CTLs', 'PDL1_CD8', 'HelperT']),
    0.8, 10, [1.0, 50.75, 100.5, 150.25, 200.0]
)
detector.process()
