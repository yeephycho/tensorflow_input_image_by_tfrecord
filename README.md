## Tensorflow tfrecord image input demo
This project demonstrates:
- How to turn your own image/label to tfrecord format.
- How to read images/labels from the generated tfrecord file.
- How to feed the images/labels to your own neural network.

Key word: [tensorflow](https://www.tensorflow.org/), input image, input label, input tfrecord

More details can be found [here](http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/).

This repo. contains two demo program, first one is a tutorial by using three python scripts to help you to manipulate on your own image dataset. The second one is a pretrained mnist model for digit recognition, you can feed your own image to the network to see the prediction result.

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
cd src
python build_image_data.py
# This operation will search the folders named after the labels in label.txt file, then turn all the files in the labeled folders to .tfrecord file. Check the label.txt file to learn more.
python read_tfrecord_data.py
# This operation will read the generated .tfrecord file into tensors/images, and write the image to the resized_image folder, the default image size is 299x299. You also can pass arguments: image_number, class_number, image_height, image_width.
```

To run the MNIST prediction:
Modify conv_mnist_inference.py file to change the image path and pre-trained model path according to your own environment.
``` bash
python conv_mnist_inference.py 
```

Command below is just a reminder to myself

git add .

git commint -m "Say whatever you want"

git push https://github.com/yeephycho/tensorflow_input_image_by_tfrecord master
