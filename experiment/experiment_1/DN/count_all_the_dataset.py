import os

import numpy as np
import pandas as pd

def get_csv_DN(dataset_path):
    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    num1 = len(train_data)

    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    num2 = len(test_data)

    print(num1)
    print(num2)
    print(num1 + num2)

def get_npz_DN(dataset_path):
    train_dataset_path = os.path.join(dataset_path, 'pku_training.npz')
    train_data = np.load(train_dataset_path, allow_pickle=True)
    x_train = train_data["words"]
    num1 = len(x_train)

    test_dataset_path = os.path.join(dataset_path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]
    num2 = len(x_test)

    print(num1)
    print(num2)
    print(num1 + num2)

if __name__ == '__main__':
    pass
