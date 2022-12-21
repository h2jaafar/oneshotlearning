import splitfolders as sf
import shutil
import os
import random
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



input_path = "./dataset/lfw2/lfw2/"

person_A_im1 = mpimg.imread(input_path+"Al_Pacino/Al_Pacino_0001.jpg")
person_A_im2 = mpimg.imread(input_path+"Al_Pacino/Al_Pacino_0003.jpg")
person_B_im1 = mpimg.imread(input_path+"Aitor_Gonzalez/Aitor_Gonzalez_0001.jpg")
person_B_im2 = mpimg.imread(input_path+"Aitor_Gonzalez/Aitor_Gonzalez_0002.jpg")

# plt.imshow(person_A_im2)

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(person_A_im1)
axs[0, 0].set_title('Person A Image 1')
axs[0, 1].imshow(person_B_im1)
axs[0, 1].set_title('Person B Image 1')
axs[1, 0].imshow(person_A_im1)
axs[1, 0].set_title('Person A Image 1')
axs[1, 1].imshow(person_A_im2)
axs[1, 1].set_title('Person A Image 2')
# for ax in axs.flat:
    # ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()