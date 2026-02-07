import numpy as np
import pandas as pd

class MeraLR:
    def __init__(self):
        self.m = None
        self.b = None
    
    def fit(self,X_train,y_train):
        X_v = X_train.values.ravel()
        y_v = y_train.values.ravel()

        X_mean = X_v.mean()
        y_mean = y_v.mean()

        num = ((X_v-X_mean)*(y_v-y_mean)).sum()
        den = ((X_v-X_mean)**2).sum()

        self.m = num/den
        self.b = y_mean-(self.m*X_mean)

    def predict(self,X_test):
        print(X_test)
        return self.m*X_test + self.b
