import dataReader
from Pipeline import trainingPipeline

if __name__ == "__main__":
    #read in data
    dr = dataReader.DataReader()

    #get data split
    train, test = dr.getTrainTestSplit()

    #train the data
    trainPipeline = trainingPipeline.TrainingPipeline(train, test)
    trainPipeline.trainModel()