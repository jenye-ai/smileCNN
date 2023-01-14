from torch.utils.data import Dataset
from torchvision import transforms

from mtcnn import MTCNN
import cv2
import cvlib as cv

class Genki4kDataset(Dataset):
    def __init__(self, labels_path, image_name_path, images_path):
        self.labels = self._load_labels(labels_path)
        self.image_names = self._load_image_names(image_name_path)
        self.images_path = images_path
        self.detector = MTCNN()

    def _load_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            labels = [line.strip()[0] for line in f]
        return labels

    def _load_image_names(self, image_name_path):
        with open(image_name_path, 'r') as f:
            image_names = [line.strip() for line in f]
        return image_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        label = self.labels[i]
        image_names= self.image_names[i]
        img_path = self.images_path + f'/{image_names}'
        image = cv2.imread(img_path)
        image_with_markers = self.create_bounding_box(image, img_path) # method call
        cropped_img = self.crop(image, image_with_markers[1])
        try:
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        except: 
            cropped_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropped_img = cv2.resize(cropped_img, (64, 64))
        cv2.normalize(cropped_img, cropped_img, 0, 255, cv2.NORM_MINMAX)
        gimage = cv2.equalizeHist(cropped_img)
        fimage = transforms.ToTensor()(gimage)
        return fimage, int(label)

    def create_bounding_box(self, image,path):
        faces = self.detector.detect_faces(image)
        
        if len(faces) < 1:
            #Use another detector here 
            face, confidences = cv.detect_face(image)

            # for f,conf in zip(face,confidences):

            #     (startX,startY) = f[0],f[1]
            #     (endX,endY) = f[2],f[3]

            #     # draw rectangle over face
            #     cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
            #     cv2.imshow("face_detection", image)

        
            # cv2.imwrite(f"/Users/jen/Documents/Code/Datasets/{path[path.rindex('/')+1:]}", image)
        
        
            chosen = confidences.index(max(confidences))
            bounding_box = face[chosen]

        else:
            bounding_box = faces[0]["box"][:4] # to obtain the only 1 image in our case

        return image, bounding_box

    def crop(self, img,bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1],bbox[0] + bbox[2], bbox[1] + bbox[3]
        bbox_obj = img[y_min:y_max, x_min:x_max]
        return bbox_obj