import numpy as np
from scipy.ndimage import rotate
import pandas as pd
import cv2

class DataGen:
    def __init__(self,csv,dir_,target_size,batch):
        self.csv = csv
        self.dir = dir_
        self.df = pd.read_csv(self.csv)
        self.batch =batch
        self.target_size = target_size
        self.images = np.array(self.df.iloc[:,0]) 
        self.kps = np.array(self.df.iloc[:,1:])
        self.length = self.images.shape[0]
        
    def rescale(self,sample):
        
        size = self.target_size[0]+50,self.target_size[1]+50
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
        
    def random_crop(self,sample):
        
        h,w = self.target_size
        image, key_pts = sample['img'], sample['kps']

        im_h, im_w = image.shape[:2]
        

        top = np.random.randint(0,im_h - h)
        left = np.random.randint(0, im_w - w)

        image = image[top: top + h,
                      left: left + w]

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
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'img': image_copy, 'kps': key_pts_copy}
        
        

    def generate(self):
        
        while True:
            batch_in=[]
            batch_out=[]
            for i in range(self.batch):
                idx = np.random.randint(self.length)
            
                img_name = self.images[idx]
                img = cv2.imread(self.dir+img_name)
                
                key_points = self.kps[idx]
                key_points = key_points.astype('float').reshape(-1, 2)
                sample = {"img": img, "kps": key_points}
                # rescaling
                sample = self.rescale(sample)
                # random_cropping
                sample = self.random_crop(sample)
                   
                 # normalizing    
                sample = self.normalize(sample)
                img = np.expand_dims(sample["img"],2)
                kps = np.reshape(sample["kps"],(136,))
                batch_in.append(img)
                batch_out.append(kps)
            # batch
            batch_in = np.array(batch_in)
            batch_out = np.array(batch_out)
            
            
            
            yield batch_in,batch_out
        