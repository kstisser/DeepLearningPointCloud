import dataReader
from Pipeline import trainingPipeline
from Visualization import visualizer

if __name__ == "__main__":
    #read in data
    dr = dataReader.DataReader()

    #get data split
    trainPillars, trainLabels, testPillars, testLabels = dr.getTrainTestSplit()

    #train the data
    trainPipeline = trainingPipeline.TrainingPipeline(trainPillars, trainLabels, testPillars, testLabels)
    results = trainPipeline.trainModel()
    dr.visualizeResults(results)