import numpy as np
import matplotlib.pyplot as plt
import time

class Motion_Model:
    def __init__(self, past_record = 10):
        '''
        Past record specifies the retentiveness (number of part record to store for ar parameter calcualtion) of model
        '''
        self.past_record = past_record
        self.parameters = {
            'ax' : np.array([[1,0,0]]),
            'ay' : np.array([[1,0,0]]),
            'varx' : np.array([50]),
            'vary' : np.array([50])
        }                                       # Parameter of position x-1, x-2, x-3 in specified order.
        self.prev_posi = np.array([])

    def add_observation(self, coordinates):
        if not (len(self.observations) == 0):
            self.prev_posi = np.append(self.prev_posi, np.array([coordinates]), 0)
        else:
            self.prev_posi = np.array([coordinates])

    def clear_data(self):
        '''Clear all previous position records'''
        self.prev_posi = np.array([])

    def update(self, current_pred):
        '''Add new observed position and update parameter'''
        if not (len(self.prev_posi) == 0):
            self.prev_posi = np.append(np.array([current_pred]), self.prev_posi, 0)
        else:
            self.prev_posi = np.array([current_pred])

        if(len(self.prev_posi) > self.past_record):
            self.prev_posi = self.prev_posi[:self.past_record, :]

        if len(self.prev_posi) >= 5:
            x_prev = np.append(np.append(self.prev_posi[0:-2, [0]], self.prev_posi[1:-1, [0]], axis=1), self.prev_posi[2:, [0]], axis=1)
            y_prev = np.append(np.append(self.prev_posi[0:-2, [1]], self.prev_posi[1:-1, [1]], axis=1), self.prev_posi[2:, [1]], axis=1)
            self.update_param(x_prev, y_prev)
            return 1
        else:
            return 0

    def find_acc(self, x):
        return x[:, [0]] - 2* x[:, [1]] + x[:, [2]]

    def update_param(self, x_prev, y_prev):
        '''Update Parameters of the model'''
        acc_x = self.find_acc(x_prev)
        acc_y = self.find_acc(y_prev)

        beta_x = float(acc_x[:-1, :].T@acc_x[1:, :] / (np.sum(acc_x[1:, :]**2)+1e-18))
        var_x = 1/(len(acc_x)-1) * np.sum((acc_x[:-1, :]- beta_x*acc_x[1:, :])**2)

        beta_y = float(acc_y[:-1, :].T@acc_y[1:, :] / (np.sum(acc_y[1:, :]**2)+1e-18))
        var_y =  1/(len(acc_y)-1) * np.sum((acc_y[:-1, :]- beta_y*acc_y[1:, :])**2)

        self.parameters['ax'] = np.array([[2+beta_x, -(1+2*beta_x), beta_x]])
        self.parameters['ay'] = np.array([[2+beta_y, -(1+2*beta_y), beta_y]])
        self.parameters['varx'] = np.array([var_x])
        self.parameters['vary'] = np.array([var_y])

    def forward(self, coordinates):
        '''Function that takes 3 position values and predicts the next position'''
        x_new = float(self.parameters['ax']@coordinates[:, [0]])
        y_new = float(self.parameters['ay']@coordinates[:, [1]])
        return [x_new, y_new]