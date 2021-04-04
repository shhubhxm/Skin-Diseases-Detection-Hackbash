# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:58:05 2021

@author: Divy
"""
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import os
import cv2
import os.path
import skimage.io as sio
from random import sample

def remove_data(folder_path,num_rem):
    files = os.listdir(folder_path)
    for file in sample(files,num_rem):
        os.remove(folder_path+file)

def augment_data(folder_path,num_files_desired):
    def random_rotation(image_array: ndarray):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-25, 25)
        return sk.transform.rotate(image_array, random_degree)

    def horizontal_flip(image_array: ndarray):
        # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
        return image_array[:, ::-1]

    def random_noise(image):
        noise_list = ["gauss","s&p","poisson","speckle"]
        noise_typ = random.choice(noise_list)
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1
            
            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy
        
    path, dirs, files = next(os.walk(folder_path))
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    num_generated_files = 0
    while num_generated_files <= num_files_desired:
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        image_to_transform = sio.imread(image_path)
        num_generated_files += 1
        
        # dictionary of the transformations functions we defined earlier
        available_transformations = {
            'rotate': random_rotation,
            'noise': random_noise,
            'horizontal_flip': horizontal_flip
            }
        
        # random num of transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1


        # define a name for our new file
        # print(image_path)
        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)
        
        # write image to the disk
        transformed_image = transformed_image.astype(np.uint8)
        sk.io.imsave(new_file_path, transformed_image)

    
    
    

parent_folder_path = "Data1H(with-noise)"
path, dirss, files = next(os.walk(parent_folder_path))
total_files_req = 800
for s in dirss:
    folder_path = f"{parent_folder_path}/{s}/"
    # print(folder_path)
    path, dirs, files = next(os.walk(folder_path))
    file_count = len(files)
    if file_count > total_files_req:
        num_rem = file_count - total_files_req
        remove_data(folder_path, num_rem)
    elif file_count < total_files_req:
        num_files_desired = total_files_req-file_count-1
        augment_data(folder_path,num_files_desired)