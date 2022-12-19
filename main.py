import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

path_separator = os.path.sep
IS_EXPERIMENT = False
train_name = 'train'
test_name = 'test'
WIDTH = HEIGHT = 105
CELLS = 1
loss_type = "binary_crossentropy"
validation_size = 0.2
early_stopping = True

from dataprep import PreProcessData
from network import SiameseNN
# LOAD_DATA = not (os.name == 'posix')
data_path = "./dataset/lfw2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MainRunner:
    def __init__(self, configs):
        print(configs)
        self.batch_size = [configs["batch_size"]]
        self.epochs = [configs["epochs"]]
        self.patience = [5]
        self.lr = [configs["lr"]]
        self.min_delta = [0.1]
        self.seeds = [0]
        self.config = configs
        self.bLoadData = configs["load_data"]
        self.run()

    def run_combined(self, l, bs, ep, pat, md, seed, train_path, test_path):
        # file types
        model_save_type = 'h5'
        # files paths
        self.initialize_seed(seed=seed)
        parameters_name = f'seed_{seed}_lr_{l}_bs_{bs}_ep_{ep}_val_{validation_size}_' \
                        f'es_{early_stopping}_pa_{pat}_md_{md}'
        print(f'Running combination with {parameters_name}')
        # A path for the weights
        load_weights_path = os.path.join(data_path, 'weights', f'weights_{parameters_name}.{model_save_type}')

        siamese = SiameseNN(config=self.config, seed=seed, img_width=WIDTH, img_height=HEIGHT, img_cells=CELLS, loss=loss_type, metrics=['accuracy'],
                                optimizer=Adam(lr=l), dropout_rate=0.4)
        siamese.fit(weights_file=load_weights_path, train_path=train_path, validation_size=validation_size,
                    batch_size=bs, epochs=ep, early_stopping=early_stopping, patience=pat,
                    min_delta=md)
        loss, accuracy = siamese.evaluate(test_file=test_path, batch_size=bs, analyze=True)
        print(f'Loss on Testing set: {loss}')
        print(f'Accuracy on Testing set: {accuracy}')
        # predict_pairs(model)
        return loss, accuracy


    def run(self):
        """
        The main function that runs the training and experiments. Uses the global variables above.
        """
        # file types
        data_set_save_type = 'pickle'
        train_path = os.path.join(data_path, f'{train_name}.{data_set_save_type}')  # A path for the train file
        test_path = os.path.join(data_path, f'{test_name}.{data_set_save_type}')  # A path for the test file
        if self.bLoadData:  # If the training data already exists
            loader = PreProcessData(img_width=WIDTH, img_height=HEIGHT, img_cells=CELLS, input_path=data_path, output_path=train_path)
            loader.load(set_name=train_name)
            loader = PreProcessData(img_width=WIDTH, img_height=HEIGHT, img_cells=CELLS, input_path=data_path, output_path=test_path)
            loader.load(set_name=test_name)

        result_path = os.path.join(data_path, f'results.csv')  # A path for the train file
        results = {'lr': [], 'batch_size': [], 'epochs': [], 'patience': [], 'min_delta': [], 'seed': [], 'loss': [],
                'accuracy': []}
        for l in self.lr:
            for bs in self.batch_size:
                for ep in self.epochs:
                    for pat in self.patience:
                        for md in self.min_delta:
                            for seed in self.seeds:
                                loss, accuracy = self.run_combined(l=l, bs=bs, ep=ep, pat=pat, md=md, seed=seed,
                                                                train_path=train_path, test_path=test_path)
                                results['lr'].append(l)
                                results['batch_size'].append(bs)
                                results['epochs'].append(ep)
                                results['patience'].append(pat)
                                results['min_delta'].append(md)
                                results['seed'].append(seed)
                                results['loss'].append(loss)
                                results['accuracy'].append(accuracy)
        df_results = pd.DataFrame.from_dict(results)
        with open(result_path, 'a') as f:
            df_results.to_csv(f)
            


    def initialize_seed(self, seed):
        """
        Initialize all relevant environments with the seed.
        """
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

