from model_package.config import *
from model_package.model_backbone import DETR

import torch
import numpy as np
import cv2


class Detector():
    def __init__(self, model_path, confidence=0.5):
        self.model = DETR(num_classes=num_classes,num_queries=num_queries)
        self.model.load_state_dict(torch.load(model_path))
        self.confidence = confidence

    @staticmethod
    def transform(image):
        '''Convert input image into object detector input image'''
        image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 1, 0)
        return image

    def run_inference_for_single_image(self, image, device):
        ''' Forward pass of detector model'''
        image = image.to(device)
        image = [image]
        self.model.eval()
        self.model.to(device)
        cpu_device = torch.device("cpu")
        
        with torch.no_grad():
            outputs = self.model(image)
            
        outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}] 
        
        y_pred = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]
        b_pred = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
        
        return y_pred, b_pred

    @staticmethod
    def scale_bbox(iw, ih, bbox):
        ''' Convert object detector output to required tracker format'''
        y1, x1, h, w = np.hsplit(bbox, 4)
        x2 = x1 + w
        y2 = y1 + h

        x1 = np.squeeze(x1 * iw)
        y1 = np.squeeze(y1 * ih)
        x2 = np.squeeze(x2 * iw)
        y2 = np.squeeze(y2 * ih)

        return np.stack((x1, y1, x2, y2), axis=-1).astype(np.int32)

    def draw(self, image, y_pred, b_pred):
        ''' Plot the detected bounding box'''
        image = np.array(image)
        for i in range(y_pred.shape[0]):
            if y_pred[i] > self.confidence:
                x1, y1, x2, y2 = b_pred[i]
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        return image

    @staticmethod
    def find_center(bound_box):
        ''' Find center of bounding box'''
        x1, y1, x2, y2 = bound_box
        return np.array([(x1+x2)/2, (y1+y2)/2])


    def find_all_centers(self, y_pred, b_pred):
        ''' Find all predicted centers of objects'''
        centers = np.array([])
        ''' Find center of the detected image '''
        for i in range(y_pred.shape[0]):
            if y_pred[i] > self.confidence:
                center = self.find_center(b_pred[i])
                if(len(centers) == 0):
                    centers = np.array([center])
                else:
                    centers = np.append(centers, np.array([center]), axis=0)
        return centers

# Debugging code
# if __name__ == '__main__':

#     args = parser.parse_args()
#     model_name = args.model
#     test_images = args.folder

#     model_path = f"saved_models/{model_name}"
#     model = DETR(num_classes=num_classes,num_queries=num_queries)
#     model.load_state_dict(torch.load(model_path))   

#     test_path = f'{test_images}/*'
#     out_path = 'samples'

#     if not os.path.exists(out_path):
#         os.mkdir(out_path)
#     confidence = 0.5

#     for i, image_path in enumerate(glob.glob(test_path)):
#         orig_image = cv2.imread(image_path)
#         h, w, _ = orig_image.shape
        
#         transformed_image = transform(orig_image)
#         y_pred, b_pred = run_inference_for_single_image(transformed_image, model=model, device=torch.device('cuda'))        
        
#         b_pred = scale_bbox(w, h, b_pred)
#         out_image = draw(orig_image, y_pred, b_pred, confidence)
#         cv2.imwrite(f'samples/{i}.jpg', out_image)        
    