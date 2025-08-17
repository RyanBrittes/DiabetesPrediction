from logisticRegression import LogisticRegression
from normalizeData import NormalizeData
from sigmoid import Sigmoid
import numpy as np
import math

logistic_regression = LogisticRegression()
normalize = NormalizeData()
sigmoid = Sigmoid()

print("Processando informações...")

results = logistic_regression.train_model()
x_test = results[3]
y_test = results[4]
trained_weight = results[0]
trained_bias = results[1]
final_loss = normalize.calc_log_denormalize_list(results[2])

for i in range(len(x_test)):
    z_test = np.array(x_test[i] @ trained_weight + trained_bias).reshape(-1, 1)
    y_predict = sigmoid.calc_sigmoid(z_test)

    print("--------------------------------------")
    print(f"Pregnancies: {math.ceil(normalize.calc_log_denormalize(x_test[i][0]))}\nGlucose: {math.ceil(normalize.calc_log_denormalize(x_test[i][1]))}\nBloodPressure: {math.ceil(normalize.calc_log_denormalize(x_test[i][2]))}\nSkinThickness: {math.ceil(normalize.calc_log_denormalize(x_test[i][3]))}\nInsulin: {math.ceil(normalize.calc_log_denormalize(x_test[i][4]))}\nBMI: {math.ceil(normalize.calc_log_denormalize(x_test[i][5]))}\nDiabetesPedigreeFunction: {normalize.calc_log_denormalize(x_test[i][6])}\nAge: {math.ceil(normalize.calc_log_denormalize(x_test[i][7]))}")
    print(f"Previsão: {np.round(y_predict)} | Real: {y_test[i]}")

print("--------------------------------------")
print(f"Pesos encontrados: {trained_weight}\nVies encontrado: {trained_bias}\nPerca final: {final_loss[-1]:4f}")