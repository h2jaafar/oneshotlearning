
import os
import pickle

import tqdm
from PIL import Image
import numpy as np


class PreProcessData(object):
  def __init__(self, img_width, img_height, img_cells, input_path, output_path):
    self.img_width = img_width
    self.img_height = img_height
    self.img_cells = img_cells
    self.input_path = input_path
    self.output_path = output_path

  def _open_image(self, path):
    img = Image.open(path)
    img = img.resize((self.img_width, self.img_height))
    return np.array(np.asarray(img), dtype='float64')

  def find_image(self, name, img_num, data_path, predict=False):
    img_num = '0'*4 + img_num # add the necessary 0's 0003
    img_num = img_num[-4:]
    img_path = os.path.join(data_path, 'lfw2', name, f'{name}_{img_num}.jpg')
    img_data = self._open_image(img_path)
    if not predict:
        img_data = img_data.reshape(self.img_width, self.img_height, self.img_cells)
    return img_data

  def find_ytf_image(self, name, img_num, data_path, predict=False):
    new_path = os.path.join(data_path, name)
    images = os.listdir(new_path)
    # print("LLLLLLLLL", int(img_num), "length", len(images))
    img = images[int(img_num)-1]
    img_path = os.path.join(new_path, img)
    # print(img_path)
    _img = Image.open(img_path).convert('L')
    # print("old img size", _img.size)
    #old_res = 301
    #new_size = 105
    _img = _img.resize((105, 105))
    # print("new img size", _img.size)
    img_data = np.array(np.asarray(_img), dtype='float64')
    if not predict:
        img_data = img_data.reshape(self.img_width, self.img_height, self.img_cells)
    return img_data

  def open_ytf(self, set_name):
    ytf_path = os.path.join("./dataset/ytf/ytf_split/")
    print("opening Youtube Faces Data")
    print("YTS: Loading training data")

    file_path = os.path.join(ytf_path, f'{set_name}.txt')
    print("Splitting data according to data in:")
    print(file_path) # train.txt or test.txt
    print("Loading data...")
    x_first = []
    x_second = []
    y = []
    names = []
    images_path = os.path.join(ytf_path, set_name)
    with open(file_path, 'r') as file:
                lines = file.readlines()
    for line in tqdm.tqdm(lines):
        line = line.split()
        if len(line) == 4:  # Class 0 - non-identical
            names.append(line)
            first_person_name, first_image_num, second_person_name, second_image_num = line[0], line[1], line[2], \
                                                                                      line[3]
            first_image = self.find_ytf_image(name=first_person_name,
                                                      img_num=first_image_num,
                                                      data_path=images_path)
            second_image = self.find_ytf_image(name=second_person_name,
                                                      img_num=second_image_num,
                                                      data_path=images_path)
            x_first.append(first_image)
            x_second.append(second_image)
            y.append(0)
        elif len(line) == 3:  # Class 1 - identical
            names.append(line)
            person_name, first_image_num, second_image_num = line[0], line[1], line[2]
            first_image = self.find_ytf_image(name=person_name,
                                                      img_num=first_image_num,
                                                      data_path=images_path)
            second_image = self.find_ytf_image(name=person_name,
                                                      img_num=second_image_num,
                                                      data_path=images_path)
            x_first.append(first_image)
            x_second.append(second_image)
            y.append(1)
        elif len(line) == 1:
            print(f'line with a single value: {line}')

    return x_first, x_second, y, names  



  def load(self, set_name):
    """
    Two classes
    Class 0: [Person A, img#, Person B, img#]
    Class 1: [Person A, img#, img#]
    """
    file_path = os.path.join(self.input_path, 'splits', f'{set_name}.txt')
    print("Splitting data according to data in:")
    print(file_path) # train.txt or test.txt
    print("Loading data...")
    x_first = []
    x_second = []
    y = []
    names = []
    with open(file_path, 'r') as file:
                lines = file.readlines()
    for line in tqdm.tqdm(lines):
        line = line.split()
        if len(line) == 4:  # Class 0 - non-identical
            names.append(line)
            first_person_name, first_image_num, second_person_name, second_image_num = line[0], line[1], line[2], \
                                                                                      line[3]
            first_image = self.find_image(name=first_person_name,
                                                      img_num=first_image_num,
                                                      data_path=self.input_path)
            second_image = self.find_image(name=second_person_name,
                                                      img_num=second_image_num,
                                                      data_path=self.input_path)
            x_first.append(first_image)
            x_second.append(second_image)
            y.append(0)
        elif len(line) == 3:  # Class 1 - identical
            names.append(line)
            person_name, first_image_num, second_image_num = line[0], line[1], line[2]
            first_image = self.find_image(name=person_name,
                                                      img_num=first_image_num,
                                                      data_path=self.input_path)
            second_image = self.find_image(name=person_name,
                                                      img_num=second_image_num,
                                                      data_path=self.input_path)
            x_first.append(first_image)
            x_second.append(second_image)
            y.append(1)
        elif len(line) == 1:
            print(f'line with a single value: {line}')
    print('Done loading dataset')
    lfw_data = [[x_first, x_second], y, names]
    x_yft_first, x_yft_second, y_yft, names_yft = self.open_ytf(set_name=set_name)
    combined_data = [[x_first+x_yft_first, x_second+x_yft_second], y + y_yft, names + names_yft]
    with open(self.output_path, 'wb') as f:
        pickle.dump(combined_data, f)

print("Loaded all data")