from FeatureModel import pointPillarFeatureNet
from ModelBackbone import pointPillarModel

class TrainingPipeline:
    def __init__(self, trainPillars, trainLabels, testPillars, testLabels):
        self.trainPillars = trainPillars
        self.trainLabels = trainLabels
        self.testPillars = testPillars
        self.testLabels = testLabels

    def trainModel(self):
        ppFeatureNet = pointPillarFeatureNet.PointPillarFeatureNet()
        ppFeatures = ppFeatureNet.feedForward()   

        ppModel = pointPillarModel.PointPillarModel("./myModel.h5py")
        ppModel.createModelBackbone(ppFeatures, self.trainPillars, self.trainLabels, self.testPillars, self.testLabels)       