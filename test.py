import torch
import torchvision
import glob
import cv2
import matplotlib.pyplot as plt

def show_images(dataset, num_samples=20, cols=4):
    """ Plots somes samples from dataset """
    plt.figure(figsize=(15,15))
    for i in range(0, num_samples):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols + 1), cols, i + 1)
        plt.imshow(dataset[i])
    plt.show()
        

file = 'resources/heightmaps/*.png'
images = [cv2.imread(image) for image in glob.glob(file)]
print(type(images[0]))
plt.imshow(images[0])
show_images(images)