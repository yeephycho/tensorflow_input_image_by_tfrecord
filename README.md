This is a demo project for convolutional neural network of tensorflow.

Key word: [tensorflow](https://www.tensorflow.org/), input image, input label, input tfrecord

This repo. contains tutorial scripts for you to easily manipulate on your own image data set, and a pretrained mnist model for digit recognition.

Many details take the official documents and tutorials as examples. But the interface is much more easier.

Prefered tensorflow version is 0.10, older version may have some problems.

The containing data set is [food101](http://food101atl.com/) dataset.

To run the prediction:
Modify conv_mnist_inference.py file to change the image path and pre-trained model path according to your own environment.
``` bash
python conv_mnist_inference.py 
```

The tensorflow I/O script is as follows:
The script serves as a tutorial to automatically turn your own image data set to tfrecord, then read the tfrecord to do the image preprocessing, you can easily modify this program to fit your own program such as training your own CNN.
``` bash
cd src
python build_image_data.py
# This operation will search the folders named after the labels in label.txt file, then turn all the files in the labeled folders to .tfrecord file.
python read_tfrecord_data.py
# This operation will read the generated .tfrecord file into tensors/images, and write the image to the resized_image folder, the image size is 299x299.
```

Just a reminder to myself

git add .

git commint -m "Say whatever you want"

git push https://github.com/yeephycho/tensorflow_input_image_by_tfrecord master
