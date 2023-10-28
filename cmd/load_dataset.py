import os
import numpy as np

from PIL import Image

PATH_ROOT = None
if os.getenv('PATH_ROOT') is not None:
    PATH_ROOT = os.getenv('PATH_ROOT')
else:
    PATH_ROOT = '/home/arthur/Documents/ifc/tc/code/sipd-mlops'

DATASET_PATH = None
if os.getenv('DATASET_PATH') is not None:
    DATASET_PATH = os.getenv('DATASET_PATH')
else:
    DATASET_PATH = '/home/arthur/Documents/dataset'


def load_dataset():
    (train_images, train_labels) = load_train_dataset()
    (test_images, test_labels) = load_test_dataset()

    np.save('./train-images.npy', train_images)
    np.save('./train-labels.npy', train_labels)

    np.save('./test-images.npy', test_images)
    np.save('./test-labels.npy', test_labels)

    return (train_images, train_labels), (test_images, test_labels) 


def load_train_dataset():
    train_dataset_path = os.path.join(DATASET_PATH, 'train')

    train_folders = os.listdir(train_dataset_path)

    images = []
    labels = []

    for train_folder in train_folders:
        if train_folder not in ['covid', 'normal', 'pneumonia']:
            continue

        cur_path = os.path.join(train_dataset_path, train_folder)

        for file in os.listdir(cur_path):
            file_path = os.path.join(train_dataset_path, train_folder, file)

            image = Image.open(file_path)
            np_image = np.asarray(image)

            images.append(np_image)

            match train_folder:
                case 'covid':
                    labels.append(0)
                case 'normal':
                    labels.append(1)
                case 'pneumonia':
                    labels.append(2)

    return np.asarray(images), np.asarray(labels)


def load_test_dataset():
    test_dataset_path = os.path.join(DATASET_PATH, 'test')

    test_folders = os.listdir(test_dataset_path)

    images = []
    labels = []

    for test_folder in test_folders:
        if test_folder not in ['covid', 'normal', 'pneumonia']:
            continue

        cur_path = os.path.join(test_dataset_path, test_folder)

        for file in os.listdir(cur_path):
            file_path = os.path.join(test_dataset_path, test_folder, file)

            image = Image.open(file_path)
            np_image = np.asarray(image)

            images.append(np_image)

            match test_folder:
                case 'covid':
                    labels.append(0)
                case 'normal':
                    labels.append(1)
                case 'pneumonia':
                    labels.append(2)
                
    return np.asarray(images), np.asarray(labels)

def shuffle_in_order(images, labels):
    assert len(images) == len(labels)  # ensure arrays are of the same length
    p = np.random.permutation(len(images))
    return images[p], labels[p]

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = load_dataset()

    (train_images, train_labels) = shuffle_in_order(train_images, train_labels)
    (test_images, test_labels) = shuffle_in_order(train_images, train_labels)

    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)