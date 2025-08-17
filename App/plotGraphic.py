from evaluateModel import EvaluateModel
from logisticRegression import LogisticRegression
from normalizeData import NormalizeData
import matplotlib.pyplot as plt

class PlotGraphic():
    def __init__(self):
        self.data_evaluated = EvaluateModel()
        self.data_trained = LogisticRegression()
        self.normalize = NormalizeData()
    
    def plot_loss(self):
        trained_parameters = self.data_trained.train_model()
        denormalize_data = self.normalize.calc_log_denormalize_list(trained_parameters[2])
        print(f'Final Loss: {denormalize_data[-1]}')

        plt.plot(denormalize_data)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss per Epochs")
        plt.grid()
        plt.show()

    def plot_confusion_matrix(self):
        evaluated_params = self.data_evaluated.calc_evaluate_model()
        params = ['TP', 'FN', 'FP', 'TN']
        value = evaluated_params[0]

        plt.bar(params, [value[0][0], value[0][1], value[1][0], value[1][1]])
        plt.title('Confusion Matrix')
        plt.xlabel('Params')
        plt.ylabel('Values')
        plt.grid()
        plt.show()
        
    def plot_metrics(self):
        evaluated_params = self.data_evaluated.calc_evaluate_model()

        metrics = ['Accuracy', 'Recall', 'FPR', 'Precision', 'F1']
        values = [evaluated_params[1], evaluated_params[2], evaluated_params[3], evaluated_params[4], evaluated_params[5]]

        plt.bar(metrics, values)
        plt.title('Evaluation Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.grid()
        plt.show()
