from FeatureModel import pointPillarFeatureNet
from ModelBackbone import pointPillarModel
from ModelBackbone import model

class TrainingPipeline:
    def __init__(self, trainPillars, trainLabels, testPillars, testLabels):
        self.trainPillars = trainPillars
        self.trainLabels = trainLabels
        self.testPillars = testPillars
        self.testLabels = testLabels

    def trainModel(self):
        '''ppFeatureNet = pointPillarFeatureNet.PointPillarFeatureNet()
        ppFeatures, input_pillars, input_indices = ppFeatureNet.feedForward()   

        ppModel = pointPillarModel.PointPillarModel("./myModel.h5py")
        ppModel.createModelBackbone(ppFeatures, self.trainPillars, self.trainLabels, self.testPillars, self.testLabels, input_pillars, input_indices)  ''' 

        mod = model.Model()
        mod.train(self.trainPillars, self.trainLabels, self.testPillars, self.testLabels)    