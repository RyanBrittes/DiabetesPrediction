from sigmoid import Sigmoid
from logLoss import LogLoss
from loadData import LoadData
import numpy as np

class LogisticRegression():
    def __init__(self):
        self.sigmoid = Sigmoid()
        self.loss = LogLoss()
        self.data = LoadData()
        self.rateTest = 0.1
        self.rateValidation = 0.1
        self.shuffledData = self.data.get_shuffle_separe_train_validation_test(self.rateTest, self.rateValidation)
        self.xTrain = self.shuffledData[0]
        self.yTrain = self.shuffledData[1]
        self.xValidation = self.shuffledData[2]
        self.yValidation = self.shuffledData[3]
        self.xTest = self.shuffledData[4]
        self.yTest = self.shuffledData[5]
        self.nSample = self.shuffledData[6]
        self.weights = np.zeros(self.shuffledData[6].shape[1])
        self.bias = 0
        self.lr = 0.00001
        self.epochs = 1000
        self.losses = []
        self.rangeProb = 0.7
        
    
    def train_model(self):

        for i in range(self.epochs):
            z_value = np.array(self.xTrain @ self.weights + self.bias).reshape(-1, 1)
            y_pred = self.sigmoid.sigmoid_calc(z_value)
            simple_loss = self.loss.calc_simple_loss(y_pred, self.yTrain)

            dw = np.array((self.xTrain.T @ simple_loss) / self.nSample).flatten()
            db = np.sum(simple_loss) / self.nSample

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self.loss.calc_log_loss(y_pred, self.yTrain)
            self.losses.append(loss)

            if(i % 100) == 0:
                print(f"Epoch: {i}\nLoss: {loss:.4f}")
    
        return [self.weights, self.bias, self.losses]
            
    def show_results(self):
        training_results = self.training_model()

        z_value = self.x @ training_results[0] + training_results[1]
        prediction = self.sigmoid.sigmoid_calc(z_value)

        calc = (prediction >= self.range_prob).astype(int)

        accuracy = np.mean(calc == self.y)

        print(f"Final Acurracy: {accuracy:.2f}")

    def values(self):
        print((self.nSample))

A = LogisticRegression()

A.show_results()


