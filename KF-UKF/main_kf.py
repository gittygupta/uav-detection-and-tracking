from model import DETR
from config import *

import cv2
import torch
import numpy as np

from kalman import kalman_filter
import inference

#Global variables to store measurements
kalman_obj = []
kalman_future = kalman_filter()

contour_pointx = []
contour_pointy = []

measured_pointx = []
measured_pointy = []

position_x = 0
position_y = 0

def main():
    model_path = 'saved_models/detr_4.pth'
    vid_name = 'Drone-Ocean.mp4'
    model = DETR(num_classes=num_classes,num_queries=num_queries)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    confidence = 0.5 

    #Framerate playback
    wait = 30

    #Number of objects to track
    obj = 1

    #Kernel size
    k = 5

    #Morph Type: 'RECT' or 'ELLIPSE'
    m_type = 'RECT'

    #Morph 'OPEN' or 'CLOSED'
    m_oc = 'CLOSED'

    if str(m_type) == 'RECT':
        m_type_cv = cv2.MORPH_RECT
    else:
        m_type_cv = cv2.MORPH_ELLIPSE
    
    if str(m_oc) == 'OPEN':
        m_oc_cv = cv2.MORPH_OPEN
    else:
        m_oc_cv = cv2.MORPH_CLOSE

    #Number of future states to predict
    future_state = 5

    cap = cv2.VideoCapture(vid_name)
    
    #Create kalman objects based on the number of future states to measure
    for i in range(0,obj):
        ka = kalman_filter()
        kalman_obj.append(ka)
    
    ############################# BACKGROUND ###############################

    # get 30 frames from the video
    avg_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

    #store frames in an array
    frames_data = []
    for i in avg_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cap.read()
        #append each frame
        frames_data.append(frame)

    # use these frames to get the average background of the video
    average_frames = np.median(frames_data, axis=0).astype(dtype=np.uint8)    

    
    #store the background
    background = cv2.cvtColor(average_frames, cv2.COLOR_BGR2GRAY)

    ####################################################################
    

    #Before beginning reset the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while (cap.isOpened()):
  
        #Get frame
        _, frame = cap.read()
        h, w, _ = frame.shape

        transformed_image = inference.transform(frame)
        y_pred, b_pred = inference.run_inference_for_single_image(transformed_image, model=model, device=torch.device('cpu'))        
        
        b_pred = inference.scale_bbox(w, h, b_pred)
        out_image = inference.draw(frame, y_pred, b_pred, confidence)
        
        #Remove the background from the frame
        #Apply a median blur for the averaging to process edges/remove noise
        background_removed = cv2.absdiff(cv2.cvtColor(cv2.medianBlur(frame,7), cv2.COLOR_BGR2GRAY), cv2.medianBlur(background,7))
        
        #Otsu's threshold to get approximate estimate of object
        _, approx_obj = cv2.threshold(background_removed, 0, 255, cv2.THRESH_OTSU)
        
        # Morphological opening operation to remove small white noise in each frame
        kernel = cv2.getStructuringElement(m_type_cv, (int(k),int(k)))
        approx_obj = cv2.morphologyEx(approx_obj, m_oc_cv, kernel)
        
        #Find Contours
        contours, hierarchy = cv2.findContours(approx_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('Contour',approx_obj)
        
        #Sort the area by the number of max contours you want
        contArea = np.array([cv2.contourArea(cont) for cont in contours])
     
        #Need to get the max contours of x objects outside a threshold
        area_contour = []

        #Getting the areas of the contours
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            area_contour.append(area)
            
        #sorting to get the max area of the contours (descending order)
        sorted_contour = sorted(zip(area_contour, contours), key=lambda x: x[0], reverse=True)
    

        #applying the kalman filter for the contours
        #loop for the number of objects which are being tracked
        for i in range(0,obj):
            if i < len(sorted_contour):
                c_x = sorted_contour[i][1]
                #Get bounding rect of the contour area
                x, y, w, h = cv2.boundingRect(c_x)

                position_x = x + w // 2
                position_y = y + h // 2

                #record the current measurement
                current_measurement = np.array([[position_x], [position_y]])
                #correct the current measurement
                kalman_obj[i].correct(current_measurement)
                #predict the next state of measurement
                current_prediction = kalman_obj[i].predict()
                
                #Position only
                xk = current_measurement[0]
                yk = current_measurement[1]
                cpx = current_prediction[0]
                cpy = current_prediction[1]
                
                

                #drawing the objects/kalman filter predictions
                cv2.circle(frame, (int(xk), int(yk)), 20, [0, 255, 0], 2, 1)
                
                #This is for single objects to showcase previous kalman prediction states (the objects trajectory)
                contour_pointx.append(int(cpx[0]))
                contour_pointy.append(int(cpy[0]))


                if obj==1:
                    for j in range(len(contour_pointx)-1):
                        x_n = contour_pointx[j]
                        y_n = contour_pointy[j]
                        
                

                ######################## Getting future kalman data from beyond 1 frame #######################################################
                cpx_future = []
                cpy_future = []
                

                cpx_future.append(cpx[0])
                cpy_future.append(cpy[0])

                #Predict the future state
                for j in range(0, future_state):
                    cpx_new = cpx_future[j]
                    cpy_new = cpy_future[j]
                    current_measurement_future = np.array([[cpx_new], [cpy_new]])
                    kalman_future.correct(current_measurement_future)
                    current_prediction_future = kalman_future.predict()
                    cpnx = current_prediction_future[0]
                    cpny = current_prediction_future[1]

                    cpx_future.append(cpnx[0])
                    cpy_future.append(cpny[0])
                    
                    cv2.circle(frame, (int(cpnx[0]), int(cpny[0])), 5, [255, 0, 0], 2, 1)

                    cv2.line(frame, (int(cpx_future[j]), int(cpy_future[j])) , (int(cpnx[0]),int(cpny[0])), (102, 255, 255), 1)

        
        #Show each frame
        cv2.imshow('Tracking',out_image)
        #quit the video on q
        key = cv2.waitKey(1)
        if cv2.waitKey(int(wait)) & 0xFF == ord('q'):
            print("Video ended.")
            break
        #pause the video on p
        if key == ord('p'):
            print("Video paused. Press any key to resume.")
            cv2.waitKey(-1) 


    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()