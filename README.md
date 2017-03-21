## Tensorflow tfrecord image input demo
This project demonstrates:
- How to turn your own image/label to tfrecord format.
- How to read images/labels from the generated tfrecord file.
- How to feed the images/labels to your own neural network.

Key word: [tensorflow](https://www.tensorflow.org/), input image, input label, input tfrecord

More details can be found [here](http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/).

This repo. contains three demo program, first one is a tutorial by using three python scripts to help you to manipulate on your own image dataset. The second one is a pretrained mnist model for digit recognition, you can feed your own image to the network to see the prediction result. The thrid demo is a deep neural network script with the tfrecord input, training and evaluation function are all added within the script, if you want to feed your own data to train a cnn I believe it's
an very good example, the script is under the src folder named flower_train_cnn.py, I trained this network from scratch, I build the very typical network structure by myself, the input comes from tensorflow imagenet finetuning project, there's five kinds of flowers. I use 290 images as test set and the rest of the images are training set, the final traning accuracy is above 97 percent and the evaluation accuracy is above 83 percent. I will document the project when
I have time, but I think if you cannot understand my source code, it may not be a very good time for you to start with tensorflow, the official tutorial mnist and cifar10 source code are all execllent tutorials.

The scrips take the official documents and tutorials as examples. The difference is that the interface is more friendly. Official tutorial only teach you how to import mnist, cifar10 or imagenet dataset, this project can help you feed your own dataset to your network.

## Dependencies
Prefered tensorflow version is 0.10, older version may have some problems.

## Dataset
The test image for image input is a subset of [food101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) dataset.

The CNN dataset is MNIST dataset.

## Usage
The procedure of using tensorflow I/O script is as follows:
The script serves as a tutorial to automatically turn your own image data set to tfrecord, then read the tfrecord to do the image preprocessing, you can easily modify this program to fit your own project to train your own CNN.
``` bash
cd ../tensorflow_input_image_by_tfrecord/src
python build_image_data.py
# This operation will search the folders named after the labels in label.txt file, then turn all the files in the labeled folders to .tfrecord file. Check the label.txt file to learn more.
python read_tfrecord_data.py --image_number=300 --class_number=3 --image_height=299 --image_width=299
# The above arguments are default.
# This operation will read the generated .tfrecord file into tensors/images, and write the image to the resized_image folder, the default image size is 299x299. You also can pass arguments: image_number, class_number, image_height, image_width.
```

To run the MNIST prediction:
Modify conv_mnist_inference.py file to change the image path and pre-trained model path according to your own environment.
``` bash
python conv_mnist_inference.py --image_path $PATH_TO_YOUR_DIGIT_IMAGE 
# e.g. python conv_mnist_inference.py --image_path=../num2.jpg
```

Command below is just a reminder to myself

git add .

git commint -m "Say whatever you want"

git push https://github.com/yeephycho/tensorflow_input_image_by_tfrecord master
