import tensorflow as tf


model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32,(5,5),input_shape=(224,224,1),data_format="channels_last"),
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
 
  
    tf.keras.layers.Flatten(),
   
 
    tf.keras.layers.Dense(1024),
    tf.keras.layers.LeakyReLU(0.1),
    
    tf.keras.layers.Dropout(0.5),
 
    tf.keras.layers.Dense(1024),
    tf.keras.layers.LeakyReLU(0.1),

   
    tf.keras.layers.Dense(136,activation="linear")
    
    
    
    
    
    
    
])