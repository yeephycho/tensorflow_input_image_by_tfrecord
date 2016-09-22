This is a demo project for convolutional neural network of tensorflow
Training dataset is MNIST dataset
Modify conv_mnist_inference.py file to change the image path and pre-trained model path according to your own environment.
``` bash
python conv_mnist_inference.py 
```
To run the prediction

The tensorflow I/O script is as follows:
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

git push https://github.com/yeephycho/tensorflow_digit_inference master
