import pandas as pd
import numpy as np
from normalizeData import NormalizeData
from dotenv import load_dotenv
import os

load_dotenv()

class LoadData():
    def __init__(self):
        self.normalize = NormalizeData()
        self.__data = pd.read_csv(os.getenv("DATAPATH"))
        self.__y_true = self.__data[['Outcome']].values
        self.__x_true = self.__data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values

    def get_dataset(self):
        return self.__data

    def get_x_value(self):
        return self.__x_true
    
    def get_y_value(self):
        return self.__y_true
    
    def get_score_Z(self, dataset):
        return self.normalize.calc_score_Z(dataset)
    
    def get_log(self, dataset):
        return self.normalize.calc_log(dataset)
    
    def get_shuffle_separe_train_validation_test(self, rateTest, rateValidation):
        xValues = self.get_score_Z(self.__x_true)
        yValues = self.get_score_Z(self.__y_true)

        np.random.seed(42)
        indexShuffled = np.random.permutation(len(yValues))

        yShuffled = yValues[indexShuffled]
        xShuffled = xValues[indexShuffled]
        
        rateTrain = 1 - rateTest - rateValidation

        lenDataset = len(yValues)
        lenTrain = np.floor(lenDataset*rateTrain).astype(int)
        lenValidation = np.round(lenDataset*rateValidation).astype(int)

        xTrain = xShuffled[0:lenTrain]
        yTrain = yShuffled[0:lenTrain]
        xValidation = xShuffled[lenTrain:(lenTrain + lenValidation)]
        yValidation = yShuffled[lenTrain:(lenTrain + lenValidation)]
        xTest = xShuffled[(lenTrain + lenValidation):lenDataset]
        yTest = yShuffled[(lenTrain + lenValidation):lenDataset]
        
        return [xTrain, yTrain, xValidation, yValidation, xTest, yTest, lenDataset]
    
A = LoadData()

value = A.get_shuffle_separe_train_validation_test(0.1, 0.1)
