import cv2
from model_package.model import Detector
import numpy as np
import time
import torch
from Particle_filter import Particle_Filter


if __name__ == "__main__":
    model_path = r'model_package/saved_models/detr_6.pth'   
    model = Detector(model_path)                                       # Initiate detector model using model_path
    cap = cv2.VideoCapture(r"Drone_Test_Set/Simulation_obstraction_diagonal.mp4")
    if(cap.isOpened() == False):
            print("Error opening video")
            quit()
    ret, frame = cap.read()

    particles = Particle_Filter(no_particles=1024,world_size=(frame.shape[1], frame.shape[0]))          # Initialise Particle Filter

    frame_no = 0
    total_time = 0.0
    model_time = 0.0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            t = time.time()
            frame_no += 1

            h, w, _ = frame.shape
            ##########################  Object Detector Estimate ############################
            transformed_image = model.transform(frame)
            y_pred, b_pred = model.run_inference_for_single_image(transformed_image, device=torch.device('cuda'))
            b_pred = model.scale_bbox(w,h,b_pred)
            out_image = model.draw(frame, y_pred, b_pred)
            frame_out = np.array(frame)
            frame_update = np.array(frame)

            with_rect = model.draw(frame, y_pred, b_pred)
            centers = model.find_all_centers(y_pred, b_pred)
            print("Time required to detect object: %.3f seconds"%(time.time()-t))
            
            ################################################################################

            #####################  Particle Filter Apply ###################################
            particles.forward()
            particles.particle_plot(frame_out)
            if(len(centers) != 0):
                with_rect = cv2.circle(with_rect, (int(centers[0][0]), int(centers[0][1])), 5, (255, 0, 0), -1)
                w = particles.weight(centers[0])
                particles.resampling(w)
            particles.particle_plot(frame_update, size = 10)
            total_time += (time.time()-t)
            print("Time required to process frame: %.3f seconds"%(time.time()-t))
            ###############################################################################

            cv2.imshow("Detector", cv2.resize(with_rect, (960,540)))
            cv2.imshow("Particles estimate", cv2.resize(frame_out, (960, 540)))
            cv2.imshow("After update", cv2.resize(frame_update, (960, 540)))
            
            
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    print("Average time to process: %.3f seconds"%(total_time/frame_no))
    cap.release()
    cv2.destroyAllWindows()