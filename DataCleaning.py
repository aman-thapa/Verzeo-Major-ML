import numpy as np
import matplotlib.pyplot as plt
from cv2 import imread, imshow
import cv2, os, time
from hashlib import md5
import scipy

import hashlib
import matplotlib.gridspec as gridspec


def function(IMAGE_DIR):
    print(IMAGE_DIR)
    os.chdir(IMAGE_DIR)
    os.getcwd()

    image_files = os.listdir()
    print(len(image_files))
    image_files[0]

    duplicates=[]
    hash_keys=dict()
    for index, filename in enumerate(os.listdir('.')):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash]=index
            else:
                duplicates.append((index,hash_keys[filehash]))
    print(len(duplicates), " Duplicates detected")
    for file_indexes in duplicates[:30]:
        try:
            plt.subplot(121)
            plt.imshow(imread(image_files[file_indexes[1]]))
            plt.title(image_files[file_indexes[1]])
            plt.xticks([])
            plt.yticks([])

            plt.subplot(122)
            plt.imshow(imread(image_files[file_indexes[0]]))
            plt.title(str(image_files[file_indexes[0]])+ 'duplicate')
            plt.xticks([])
            plt.yticks([])
            plt.show()

        except OSError as e:
            continue

    for index in duplicates:
        os.remove(image_files[index[0]])

if __name__ == '__main__':
    function('./')
