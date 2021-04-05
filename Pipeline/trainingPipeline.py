from FeatureModel import pointPillarFeatureNet
from ModelBackbone import pointPillarModel

class TrainingPipeline:
    def __init__(self, trainData, testData, params=None):
        self.trainingData = trainData
        self.testData = testData
        self.params = params

    def trainModel(self):
        ppFeatureNet = pointPillarFeatureNet.PointPillarFeatureNet(self.trainingData)
        ppFeatures = ppFeatureNet.feedForward()   

        ppModel = pointPillarModel.PointPillarModel(params, "./model.h5py")
        ppModel.createModelBackbone(ppFeatures)       