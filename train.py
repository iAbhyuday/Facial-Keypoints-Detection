import tensorflow as tf
import numpy as np
import matplolib.pyplot as plt
from utils.ImageGenerator import DataGen

# Limiting Tensorflow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
   # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# Helper function for visualising key-points
def showKeypoints(img,kps):
    plt.imshow(img)
    kps = np.reshape(kps,(-1,2))
    plt.scatter(kps[:,0],kps[:,1],s=4,c='red')
    plt.show()

# Learning rate scheduler
def scheduler(epoch):
    
    if epoch>60:
        return 0.001*tf.math.exp(0.1*(60-60-epoch))
    else:
        return 0.001

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train data generator with batch_size=64 and target_size =(96,96)
train = DataGen("train/train.csv","train/training_images/",(96,96),64) 
train_gen = train.generate()


# LeNet 

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32,(5,5),input_shape=(96,96,1),data_format="channels_last"),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D((2,2),2),
    
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.Conv2D(64,(3,3)),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D((2,2),2),
    
    tf.keras.layers.Dropout(0.2),
   
    tf.keras.layers.Conv2D(128,(3,3)),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D((2,2),2),
    
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(256,(3,3)),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D((2,2),2),
    
    tf.keras.layers.Dropout(0.4),
 
    tf.keras.layers.Conv2D(512,(1,1)),
    tf.keras.layers.LeakyReLU(0.1),
    tf.keras.layers.MaxPool2D((2,2),2),
    
  
    tf.keras.layers.Flatten(),
   
 
    tf.keras.layers.Dense(1024),
    tf.keras.layers.LeakyReLU(0.1),
    
    tf.keras.layers.Dropout(0.5),
 
    tf.keras.layers.Dense(1024),
    tf.keras.layers.LeakyReLU(0.1),

   
    tf.keras.layers.Dense(136,activation="linear")
    
    
])


# model.compile(optimizer = "adam",loss = "mse", metric = ["acc"])

# model.fit(train_gen,epochs=100,callbacks = [callback])

# model.save("saved_models/model")
# print("Saved model in saved_models/ ")