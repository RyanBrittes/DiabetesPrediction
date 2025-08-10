from sigmoid import Sigmoid
from logLoss import LogLoss
from loadData import LoadData
import numpy as np

class LogisticRegression():
    def __init__(self):
        self.sigmoid = Sigmoid()
        self.loss = LogLoss()
        self.data = LoadData()
        self.rate_test = 0.1
        self.rate_validation = 0.1
        self.shuffled_data = self.data.get_shuffle_separe_train_validation_test(self.rate_test, self.rate_validation)
        self.x_train = self.shuffled_data[0]
        self.y_train = self.shuffled_data[1]
        self.x_validation = self.shuffled_data[2]
        self.y_validation = self.shuffled_data[3]
        self.x_test = self.shuffled_data[4]
        self.y_test = self.shuffled_data[5]
        self.len_sample = self.shuffled_data[6]
        self.weights = np.zeros(self.shuffled_data[0].shape[1])
        self.bias = 0
        self.lr = 0.00001
        self.epochs = 1000
        self.losses = []
        self.threshold = 0.7
    
    def train_model(self):

        for i in range(self.epochs):
            z_value = np.array(self.x_train @ self.weights + self.bias).reshape(-1, 1)
            y_pred = self.sigmoid.calc_sigmoid(z_value)
            simple_loss = self.loss.calc_simple_loss(y_pred, self.y_train)

            dw = np.array((self.x_train.T @ simple_loss) / self.len_sample).flatten()
            db = np.sum(simple_loss) / self.len_sample

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self.loss.calc_log_loss(y_pred, self.y_train)
            self.losses.append(loss)

            #if(i % 100) == 0:
                #print(f"Epoch: {i}\nLoss: {loss:.4f}")
    
        return [self.weights, self.bias, self.losses, self.x_test, self.y_test]

