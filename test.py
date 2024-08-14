import torch
import torchvision
import glob
import cv2
import matplotlib.pyploy as plt

def show_images(dataset, num_samples=20, cols=4):
    """ Plots somes samples from dataset """
    plt.figure(figsize=(15,15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplt(num_samples/cols + 1, cols, i + 1)
        plt.imshow(img[0])

