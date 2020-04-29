import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2

class DataGen(tf.keras.utils.Sequence):
    def __init__(self,csv,dir_,target_size,batch_size):
        self.csv = csv
        self.dir = dir_
        self.batch_size = batch_size
        self.target_size = target_size
        self.df = pd.read_csv(self.csv)
        
        self.images = np.array(self.df.iloc[:,0])
        self.indexes = np.arange(0,self.images.shape[0]) 
        self.kps = np.array(self.df.iloc[:,1:])
        self.length = self.images.shape[0]
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.length / self.batch_size))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        imgs = self.images[index*self.batch_size:(index+1)*self.batch_size]
        kps = self.kps[index*self.batch_size:(index+1)*self.batch_size]

       
        # Generate data
        X, y = self.__data_generation(imgs,kps)

        return X, y
        
    def rescale(self,sample):
        #if size is None:
        size = self.target_size[0],self.target_size[1]
        img = sample["img"]
        kps = sample["kps"]
        
        h,w = img.shape[:2]
        if isinstance(size, int):
            if h > w:
                new_h, new_w = size[0] * h / w, size[1]
            else:
                new_h, new_w = size[0], size[1] * w / h
        else:
            new_h,new_w = size
        new_h,new_w = int(new_h),int(new_w)
        new_im = cv2.resize(img,(new_h,new_w))
        new_kps = kps * [new_w / w, new_h / h]
        
        return {"img": new_im,"kps":new_kps}
        
    def random_crop(self,sample):
        size = self.target_size[0]
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
        
        
    def on_epoch_end(self):
      pass
        
        
    def __data_generation(self,imgs,kps):
        
        
        batch_imgs=[]
        batch_kps =[]
        for i, ID in enumerate(imgs):

          prob =0.5
            # rescaling
            
          img = cv2.imread(self.dir+ID)
          key_points = kps[i]
          key_points = key_points.astype('float').reshape(-1, 2)
          sample = {"img": img, "kps": key_points}
           
            
          if(prob<np.random.rand()):
                
                    
            sample = self.rescale(sample,crop=True)
            sample = self.random_crop(sample)
                    
          else:

            sample = self.rescale(sample,crop=False)
          sample = self.normalize(sample)
          batch_imgs.append(np.expand_dims(sample["img"],axis=2))
          batch_kps.append(sample["kps"].reshape(136,))
        batch_imgs = np.array(batch_imgs)
        batch_kps = np.array(batch_kps)
            
            
            
        return batch_imgs,batch_kps
        