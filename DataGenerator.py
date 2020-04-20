import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import tensorflow.keras.layers as l
%matplotlib inline


class DataGen:
    def __init__(self,csv,dir_):
        self.csv = csv
        self.dir = dir_
        self.df = pd.read_csv(self.csv)
        
        self.images = np.array(self.df.iloc[:,0]) 
        self.kps = np.array(self.df.iloc[:,1:])
        self.length = self.images.shape[0]
        
    def rescale(self,sample,size):
        assert isinstance(size, (int, tuple))
        img = sample["img"]
        kps = sample["kps"]
        
        h,w = img.shape[:2]
        if isinstance(size, int):
            if h > w:
                new_h, new_w = size * h / w, size
            else:
                new_h, new_w = size, size * w / h
        else:
            new_h,new_w = size
        new_h,new_w = int(new_h),int(new_w)
        new_im = cv2.resize(img,(new_h,new_w))
        new_kps = kps * [new_w / w, new_h / h]
        
        return {"img": new_im,"kps":new_kps}
        
    def random_crop(self,sample,size):
        
        image, key_pts = sample['img'], sample['kps']

        im_h, im_w = image.shape[:2]
        

        top = np.random.randint(0, im_h - size)
        left = np.random.randint(0, im_w - size)

        image = image[top: top + size,
                      left: left + size]

        key_pts = key_pts - [left, top]

        return {'img': image, 'kps': key_pts}
    
    def normalize(self, sample):
        image, key_pts = sample['img'], sample['kps']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'img': image_copy, 'kps': key_pts_copy}
        
        

        
        
    def generate(self):
        
        while True:
            
            idx = np.random.randint(self.length)
            
            img_name = self.images[idx]
            img = cv2.imread(self.dir+img_name)
            
            prob =0.5
            # rescaling
            
            
            key_points = self.kps[idx]
            key_points = key_points.astype('float').reshape(-1, 2)
            sample = {"img": img, "kps": key_points}
           
            
            if(prob<np.random.rand()):
                
                sample = self.rescale(sample,(250,250))
                sample = self.random_crop(sample,224)
            else:
                 sample = self.rescale(sample,(96,96))
            sample = self.normalize(sample)
            
            
            
            yield sample
        