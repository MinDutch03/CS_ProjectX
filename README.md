# Project: Gender and Age Prediction 🏹

# 1. Introduction

# 1.1   Goal

> To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.

# 1.2   About the Project

* In this particular practical I create project **Gender and Age Prediction.**
* For Age and Gender I used some pre-trained model which you can find in **modelNweight** folder.
* You can also try it on your image.
* Age is Categorized in Following Categories:
  * (0-6)
  * (7-13)
  * (14-19)
  * (20-29)
  * (30-39)
  * (40-55)
  * (56-60)
  * (61-70)
  * (71-100)
* It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression

# 1.3  Dataset

For this python project, I used the Adience dataset; the dataset is available in the public domain and you can find it [here](https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification). This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used was trained on this dataset. By virtue of its large size, I only kept 3 folders of pictures.

To download the dataset on Kaggle, I showed you in *downlod_data.py*  file.

# 2.  Install Dependencies 📥

```bash
pip install opencv-python
pip install matplotlib
pip install opendatasets
pip install pillow
```

# 3. Main Credits For Pretrained Models 🌟:

- [faceProto](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt)
- [faceModel](https://github.com/spmallick/learnopencv/blob/master/AgeGender/opencv_face_detector_uint8.pb)
- [ageProto](https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_deploy.prototxt)
- [ageModel](https://github.com/GilLevi/AgeGenderDeepLearning/blob/master/models/age_net.caffemodel)
- [genderProto](https://github.com/spmallick/learnopencv/blob/master/AgeGender/gender_deploy.prototxt)
- [genderModel](https://github.com/eveningglow/age-and-gender-classification/blob/master/model/gender_net.caffemodel)

  For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

# 4. Usage

* Download my Repository
* Open your Command Prompt or Terminal and change directory to the folder where all the files are present.
* **Detecting Gender and Age of face in Image** Use Command :

  ```
    py gad.py
  ```

  Note: The Image should be present in same folder where all the files are present

# Prerequisites 🚀

* Python
* OpenCV
* Matplotlib
* Pillow

# Documentation 🎯

* [Pandas](https://pandas.pydata.org/docs/)
* [OpenCV](https://opencv.org/)
* [Matplotlib](https://matplotlib.org/stable/contents.html)
* [Pillow](https://pillow.readthedocs.io/en/stable/index.html)
