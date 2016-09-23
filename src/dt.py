import os  # handle system path and filenames
import tensorflow as tf  # import tensorflow as usual

# define a function to list tfrecord files.
def list_tfrecord_file(file_list):
    tfrecord_list = []
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(file_list[i])		
        if current_file_abs_path.endswith(".tfrecord"):
            tfrecord_list.append(current_file_abs_path)
            print("Found %s successfully!" % file_list[i])				
        else:
            pass
    return tfrecord_list
	
# Traverse current directory
current_folder_filename_list = os.listdir("./")
if current_folder_filename_list != None:
    print("%s files were found under current folder. " % len(current_folder_filename_list))
    print("Please be noted that only files end with '*.tfrecord' will be load!")
    tfrecord_list = list_tfrecord_file(current_folder_filename_list)
    if len(tfrecord_list) != 0:
        print(tfrecord_list)
    else:
        print("Cannot find any tfrecord files, please check the path.")
