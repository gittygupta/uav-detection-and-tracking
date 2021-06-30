from model import DETR
from inference import *
from config import *

import cv2
import time
import torch
import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y, std_acc, x_std_meas, y_std_meas):
        
        #x_std_meas ans y_std_meas are standard diviation about x and y axis
        self.dt = dt

        #Input variables in 2D  
        self.u = np.matrix([[u_x],[u_y]])

        ### Initialisation ###
        self.x = np.matrix([[0], [0], [0], [0]])

        self.F = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.G = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        self.P = np.eye(self.F.shape[1])

    def predict(self):
        self.x = np.dot(self.F, self.x) + np.dot(self.G, self.u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P
        return self.x[0:2]

def find_center(bound_box):
    x1, y1, x2, y2 = bound_box
    return np.array([(x1+x2)/2, (y1+y2)/2])


def find_all_centers(y_pred, b_pred, confidence):
    centers = np.array([])
    ''' Find center of the detected image '''
    for i in range(y_pred.shape[0]):
        if y_pred[i] > confidence:
            center = find_center(b_pred[i])
            if(len(centers) == 0):
                centers = np.array([center])
            else:
                centers = np.append(centers, np.array([center]), axis=0)
    return centers

if __name__ == "__main__":
    model_name = 'detr_6.pth'
    model_path = f"saved_models/{model_name}"
    model = DETR(num_classes=num_classes,num_queries=num_queries)
    model.load_state_dict(torch.load(model_path)) 
    
    cap = cv2.VideoCapture("Input_video.mp4")
    if(cap.isOpened() == False):
        print("Error opening video")

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
    
    confidence = 0.5
    frame_no = 0
    out = cv2.VideoWriter('output_videos/0.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         20, (960, 540))
                         
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        frame_no += 1

        transformed_frame = transform(frame)
        y_pred, b_pred = run_inference_for_single_image(transformed_frame, model, torch.device('cuda'))
        b_pred = scale_bbox(frame.shape[1], frame.shape[0], b_pred)
        bbox_image = draw(frame, y_pred, b_pred, confidence)

        centers = find_all_centers(y_pred, b_pred, confidence)

        if (len(centers) > 0):
            centers = np.reshape(centers, (centers.shape[1], centers.shape[0]))
            x, y = KF.predict()
            x1, y1 = KF.update(centers)

            #Mark the estimated position in the video
            cv2.circle(frame, (int(x1), int(y1)), 8, (0, 0, 255), -1)

        out.write(cv2.resize(frame, (960, 540)))
        #cv2.imshow('output', cv2.resize(frame, (960, 540)))
        #cv2.imshow('bboxes', cv2.resize(bbox_image, (960, 540)))

        print(f'Frame: {frame_no}, total time: {time.time() - start_time}')

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    out.release()
    #cv2.destroyAllWindows()
