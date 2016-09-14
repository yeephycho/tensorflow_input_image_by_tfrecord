import os

import tensorflow as tf

from scipy import misc

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def convert_to(image, labels, name = None):
    
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]
    
    filename = "yixuan.tfrecords"
    print("Writing", filename)
    writer = tf.python_io.TFRecordWriter(filename)

    image_raw = image.tostring()
    example = tf.train.Example(features = tf.train.Features(feature = {
        "height": _int64_feature(rows),
        "width": _int64_feature(cols),
        "depth": _int64_feature(depth),
        "label": _int64_feature(0),
        "image": _bytes_feature(image_raw)
        }))
    writer.write(example.SerializeToString())



# Use this function to read data, currently misc from scipy is used.

def decode_image_file(file_list):
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(file_list[i])
        print("Now checking the following file: %s" % current_file_abs_path)

        if current_file_abs_path.endswith(".jpg"):
            current_image = misc.imread(current_file_abs_path)
            if(current_image != None):
                convert_to(current_image, 0, None)
                print("Import image successfully!")
            else:
                print("Unexpected error when calling misc.imread()")

        elif current_file_abs_path.endswith(".jpeg"):
            current_image = misc.imread(current_file_abs_path)
            if(current_image != None):
                print("Import image successfully!")
            else:
                print("Unexpected error when calling misc.imread()")

        elif current_file_abs_path.endswith(".png"):
            current_image = misc.imread(current_file_abs_path)
            if(current_image != None):
                print("Import image successfully!")
            else:
                print("Unexpected error when calling misc.imread()")

        else:
            print("Only support image files matches the pattern of '*.jpeg', '*.jpg', '*.png', so current file %s will not be processed." % current_file_abs_path)
    return True


# Traverse a directory

current_folder_filename_list = os.listdir("./")
if current_folder_filename_list != None:
    print("Current folder has the following files: %s. " % current_folder_filename_list)
    decode_image_file(current_folder_filename_list)
else:
    print("Cannot find any files, please check the path.")

