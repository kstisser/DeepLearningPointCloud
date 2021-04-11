import dataReader
from Pipeline import trainingPipeline

if __name__ == "__main__":
    #read in data
    dr = dataReader.DataReader(readingDummyData = True)

    #get data split
    trainPillars, trainLabels, testPillars, testLabels = dr.getTrainTestSplit()

    #train the data
    trainPipeline = trainingPipeline.TrainingPipeline(trainPillars, trainLabels, testPillars, testLabels)
    trainPipeline.trainModel()