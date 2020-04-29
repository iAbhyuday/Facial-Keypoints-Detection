<div align="center"><img src="https://idroot.us/wp-content/uploads/2019/03/TensorFlow-logo.png" height= "90px" width="155px"></div>
<div align="center"><h3>Facial Keypoints Detection</h3></div>

<br><br>


### Description
This project uses LeNet inspired CNN architecture to detect 68 facial keypoints in an image. Facial keypoints include points around eyes, noses, eyebrows, mouth and facial border. 
Facial Keypoints have applications in face morphing, emotion recognition, face-tracking, face-pose recognition, facial filters and many more... 

#### Requirements
>* Tensorflow
>* OpenCV
>* skimage


### Dataset

Youtube faces dataset is used for training/testing and can be accesed from [here](https://github.com/udacity/P1_Facial_Keypoints/tree/master/data). 
There are 3,462 training images and around 2000 test images.
Augmentations used : Random Crop and 5 degrees clockwise, anticlockwise rotation

### Results 
| | | 
|:----------------------:|:---------------------:|
|<img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/test_images/Robert.jpg"> |  <img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/results/robert.jpg">|
|<img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/test_images/michael%20jackson.jpg">  |  <img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/results/mj.jpg">|
|<img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/test_images/elon.jpg">  |  <img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/results/elon.jpg">|
|<img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/test_images/leo.jpg">|<img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/results/leo.jpg">|
|<img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/test_images/oscars.jpg">|<img width="500" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/iAbhyuday/Facial-Keypoints-Detection/raw/master/results/oscars.jpg">|
