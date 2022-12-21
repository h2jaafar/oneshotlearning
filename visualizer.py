import splitfolders as sf
import shutil
import os
import random
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


input_path = "./dataset/lfw2/lfw2/"
bjarne = "./docs/bjarne.jpeg"
person_A_im1 = mpimg.imread(input_path+"Al_Pacino/Al_Pacino_0001.jpg")
person_A_im2 = mpimg.imread(input_path+"Al_Pacino/Al_Pacino_0003.jpg")
person_B_im1 = mpimg.imread(input_path+"Aitor_Gonzalez/Aitor_Gonzalez_0001.jpg")
# person_B_im2 = mpimg.imread(input_path+"Aitor_Gonzalez/Aitor_Gonzalez_0002.jpg")
person_B_im2 = mpimg.imread(bjarne)

image_2 = Image.open("./docs/bjarne_.jpg").convert("L")
image_2 = image_2.transpose(Image.FLIP_LEFT_RIGHT)
image = Image.open(bjarne).convert("L")
image_2.resize((105,105))
image.resize((105,105))
arr2 = np.asarray(image_2)
arr1 = np.asarray(image)

# plt.imshow(person_A_im2)

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(arr1, cmap="gray", vmin=0, vmax=255)
axs[0, 0].set_title('Person A Image 1')
axs[0, 1].imshow(person_A_im1, cmap="gray", vmin=0, vmax=255)
axs[0, 1].set_title('Person B Image 1')
axs[1, 0].imshow(arr1, cmap="gray", vmin=0, vmax=255)
axs[1, 0].set_title('Person A Image 1')
axs[1, 1].imshow(arr2, cmap="gray", vmin=0, vmax=255)
axs[1, 1].set_title('Person A Image 2')
# for ax in axs.flat:
    # ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()