import splitfolders as sf
import shutil
import os
import random
import tqdm
def extractImgs():
  source = "./dataset/ytf/ytf/aligned_images_DB/"
  destination = "./dataset/ytf/ytf/aligned_images_DB/"
  files_list = os.listdir(source)
  prog = 0
  for files in files_list:
      person = os.path.join(source, files)
      # print(person)
      person_files = os.listdir(person)
      prog = prog + 1
      print(prog)
      for folder in person_files:
        file_path = os.path.join(person, folder)
        for file in os.listdir(file_path):
          new_path = './dataset/ytf/ytf/imgs/'
          new_path = os.path.join(new_path, files+"/")
          full_path = file_path + "/" +  file
          print("moving from", full_path, "to ", new_path)
          # print("NEw path", new_path)
          if not os.path.exists(new_path):
            os.mkdir(new_path)
          shutil.copy(full_path, new_path)
        shutil.copytree(file, person)
        
      # individual_list = os.listdir(curr)
      # for individual in individual_list:
      #   i = os.path.join(source,files, individual)
      #   imgs = os.listdir(i)
      #   for img in imgs: 
      #     # print(img)
      #     print("img", img, "moved up to ", curr)
      #     shutil.move(img, curr)

  """
  Script to split the ytf dataset into train, validation and test. 
  """





def split():
  input_path = "./dataset/ytf/ytf/imgs/"
  output_path = "./dataset/ytf/ytf_split/"

  sf.ratio(input=input_path, output=output_path, seed = 1337, ratio = (.8, .1, .1))

def createSplitsFile():

  # train:
  train_path = "./dataset/ytf/ytf_split/train/"
  files = os.listdir(train_path)
  len_files = len(files)
  # files = files[:-0.2*len_files] # remove 20% of the list
  with open('./dataset/ytf/ytf_split/train.txt', 'w') as f:
    f.write(str(len_files) + "\n")
  for person in tqdm.tqdm(files):
    list_files = os.listdir(os.path.join(train_path, person))
    length = len(list_files)
    img_num = random.randint(0, length)
    img_num_2 = random.randint(0, length)
    person_ran_num = random.randint(1,len_files-1)
    other_person = files[person_ran_num]
    if other_person!=person: # class 0, diff people
      length_other_person = len(os.listdir(os.path.join(train_path, other_person)))
      other_img_num = random.randint(0, length_other_person)
      with open('./dataset/ytf/ytf_split/train.txt', 'a') as f:
        list_to_write = str(person) + "\t" + str(img_num) + "\t" + str(other_person) + "\t" + str(other_img_num) + "\n"
        list_to_write_same = str(person) + "\t" +  str(img_num) + "\t" + str(img_num_2) + "\n"
        f.write(list_to_write)
        f.write(list_to_write_same)

  # test:
  train_path = "./dataset/ytf/ytf_split/test/"
  files = os.listdir(train_path)
  len_files = len(files)
  # files = files[:-0.2*len_files] # remove 20% of the list
  with open('./dataset/ytf/ytf_split/test.txt', 'w') as f:
    f.write(str(len_files) + "\n")
  for person in tqdm.tqdm(files):
    list_files = os.listdir(os.path.join(train_path, person))
    length = len(list_files)
    img_num = random.randint(0, length)
    img_num_2 = random.randint(0, length)
    person_ran_num = random.randint(1,len_files-1)
    other_person = files[person_ran_num]
    if other_person!=person: # class 0, diff people
      length_other_person = len(os.listdir(os.path.join(train_path, other_person)))
      other_img_num = random.randint(0, length_other_person)
      with open('./dataset/ytf/ytf_split/test.txt', 'a') as f:
        list_to_write = str(person) + "\t" + str(img_num) + "\t" + str(other_person) + "\t" + str(other_img_num) + "\n" # class 0 (Diff)
        list_to_write_same = str(person) + "\t" +  str(img_num) + "\t" + str(img_num_2) + "\n" # class 1 (same)
        f.write(list_to_write)
        f.write(list_to_write_same)



createSplitsFile()