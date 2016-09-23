This is a demo convolutional neural network project for image digit recognition by tensorflow.

Key word: [tensorflow](https://www.tensorflow.org/), input image, input label, input tfrecord

More details can be found in post [here](http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/).

This repo. contains two program, first one is a tutorial by three python scripts to help you to manipulate on your own image dataset with very small effort, and the second one is a pretrained mnist model for digit recognition, you can feed your own image to the network to see the performance.

Many details of the scrips take the official documents and tutorials as examples. But the interface is much more easier, the official tutorial only teach you how to import mnist, cifar10 or imagenet dataset, this project can help you feed your own image data to your network.

Prefered tensorflow version is 0.10, older version may have some problems.

The test image data is a subset of [food101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) dataset.

To run the prediction:
Modify conv_mnist_inference.py file to change the image path and pre-trained model path according to your own environment.
``` bash
python conv_mnist_inference.py 
```

The procedure of using tensorflow I/O script is as follows:
The script serves as a tutorial to automatically turn your own image data set to tfrecord, then read the tfrecord to do the image preprocessing, you can easily modify this program to fit your own project to train your own CNN.
``` bash
cd src
python build_image_data.py
# This operation will search the folders named after the labels in label.txt file, then turn all the files in the labeled folders to .tfrecord file. Check the label.txt file to learn more.
python read_tfrecord_data.py
# This operation will read the generated .tfrecord file into tensors/images, and write the image to the resized_image folder, the default image size is 299x299. You also can pass arguments: image_number, class_number, image_height, image_width.
```

Command below is just a reminder to myself

git add .

git commint -m "Say whatever you want"

git push https://github.com/yeephycho/tensorflow_input_image_by_tfrecord master
