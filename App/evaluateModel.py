from logisticRegression import LogisticRegression
from sigmoid import Sigmoid
import numpy as np

class EvaluateModel():
    def __init__(self):
        self.logistic_regression = LogisticRegression()
        self.sigmoid = Sigmoid()
        self.data_train = self.logistic_regression.train_model()
        self.threshold = 0.5

    def calc_prediction(self):
        training_results = self.logistic_regression.train_model()

        z_value = training_results[3] @ training_results[0] + training_results[1]
        y_predict = self.sigmoid.calc_sigmoid(z_value)

        y_predict_binary = (y_predict >= self.threshold).astype(int)

        return [y_predict_binary, np.array(training_results[4]).flatten()]

    def calc_evaluate_model(self):
        y_value = self.calc_prediction()

        tp_value = int(np.sum((y_value[0] == 1) & (y_value[1] == 1)))
        tn_value = int(np.sum((y_value[0] == 0) & (y_value[1] == 0)))
        fp_value = int(np.sum((y_value[0] == 1) & (y_value[1] == 0)))
        fn_value = int(np.sum((y_value[0] == 0) & (y_value[1] == 1)))
        
        matriz_confusion = [[tp_value, fn_value], [fp_value, tn_value]]

        accuracy = ((tp_value + tn_value) / (tp_value + tn_value + fp_value + fn_value)) * 100
        recall = (tp_value / (tp_value + fn_value)) * 100 if (tp_value + fn_value) > 0 else 0
        fpr = (fp_value / (fp_value + tn_value)) * 100 if (tp_value + tn_value) > 0 else 0
        precision = (tp_value / (tp_value + fp_value)) * 100 if (tp_value + fp_value) > 0 else 0
        f1 = 2 * (recall * precision) / (precision + recall) if (precision + recall) > 0 else 0

        return [matriz_confusion, accuracy, recall, fpr, precision, f1]
