import numpy as np

class NormalizeData():

    def calc_mean(self, rawValue):
        sumValue = 0
        for i in range(len(rawValue)):
            sumValue += rawValue[i]

        return sumValue/len(rawValue)
    
    def calc_standard_deviation(self, rawValue):
        mean = self.calc_mean(rawValue)
        sumValue = 0
        for i in range(len(rawValue)):
            sumValue += (rawValue[i] - mean) ** 2
        
        return (sumValue / len(rawValue)) ** 0.5
    
    def calc_score_Z(self, rawValue):
        meanValue = self.calc_mean(rawValue)
        stdValue = self.calc_standard_deviation(rawValue)
        listValue = []

        for i in range(len(rawValue)):
            value = (rawValue[i] - meanValue) / stdValue
            listValue.append(value)
        
        return np.vstack(listValue)
    
    def calc_log(self, rawValue):
        listValue = []
        const_not_zero = 1e-8
        for i in range(len(rawValue)):
            listValue.append(np.log(rawValue[i] + const_not_zero))
        return np.vstack(listValue)
    
    def calc_log_denormalize(self, rawValue):
        return np.exp(rawValue)