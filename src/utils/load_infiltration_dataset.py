import os
import cv2
import numpy as np

from tqdm import tqdm
from PIL import Image

PATH_ROOT = os.getenv('PATH_ROOT')
DATASET2_PATH = os.getenv('DATASET2_PATH')
TEST_DATASET_PATH = os.getenv('TEST_DATASET_PATH')
TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')

def load_dataset():
    if already_exists():
        (train_images, train_labels), (test_images, test_labels) = load_all_dataset()
    else:
        (test_images, test_labels) = load_test_dataset()
        (train_images, train_labels) = load_train_dataset()
        save_dataset((train_images, train_labels), (test_images, test_labels))

    return (train_images, train_labels), (test_images, test_labels) 

def load_all_dataset():
    # load default train dataset
     train_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_infiltration_images.npy'))
     train_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_infiltration_labels.npy'))
    # load default test dataset
     test_images = np.load(os.path.join(TEST_DATASET_PATH, 'original', 'test_infiltration_images.npy'))
     test_labels = np.load(os.path.join(TEST_DATASET_PATH, 'original', 'test_infiltration_labels.npy'))

     return (train_images, train_labels), (test_images, test_labels)

def save_dataset(train, test):
    (test_images, test_labels) = test
    (train_images, train_labels) = train

    # save default train dataset
    np.save(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_infiltration_images.npy'), train_images)
    np.save(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_infiltration_labels.npy'), train_labels)
    # save default test dataset
    np.save(os.path.join(TEST_DATASET_PATH, 'original', 'test_infiltration_images.npy'), test_images)
    np.save(os.path.join(TEST_DATASET_PATH, 'original', 'test_infiltration_labels.npy'), test_labels)

def already_exists():
    train_images_exists = os.path.exists(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_infiltration_images.npy'))
    train_labels_exists = os.path.exists(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_infiltration_labels.npy'))
    test_images_exists = os.path.exists(os.path.join(TEST_DATASET_PATH, 'original', 'test_infiltration_images.npy'))
    test_labels_exists = os.path.exists(os.path.join(TEST_DATASET_PATH, 'original', 'test_infiltration_labels.npy'))

    if train_images_exists and train_labels_exists and test_images_exists and test_labels_exists:
        return True

    return False


def load_train_dataset():
    train_dataset_path = os.path.join(DATASET2_PATH, 'train')

    images = []
    labels = []

    for file in tqdm(os.listdir(train_dataset_path), desc=f"Lendo treinamento"):
        file_path = os.path.join(train_dataset_path, file)

        image = cv2.imread(file_path)
        if image is None:
          # print some err
          continue

        image = cv2.resize(image, (224, 224))
        np_image = np.asarray(image, dtype=np.uint8)

        images.append(np_image)
        labels.append(3)

    return np.asarray(images), np.asarray(labels)


def load_test_dataset():
    test_dataset_path = os.path.join(DATASET2_PATH, 'test')

    images = []
    labels = []

    for file in tqdm(os.listdir(test_dataset_path), desc=f"Lendo teste"):
        file_path = os.path.join(test_dataset_path, file)

        image = cv2.imread(file_path)
        if image is None:
          # print some err
          continue

        image = cv2.resize(image, (224, 224))
        np_image = np.asarray(image, dtype=np.uint8)

        images.append(np_image)
        labels.append(3)
            
    return np.asarray(images), np.asarray(labels)

def shuffle_in_order(images, labels):
    print('len(images): ', images.shape)
    print('len(labels): ', labels.shape)

    assert len(images) == len(labels)  # ensure arrays are of the same length
    p = np.random.permutation(len(images))
    return images[p], labels[p]

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = load_dataset()

    (train_images, train_labels) = shuffle_in_order(train_images, train_labels)
    (test_images, test_labels) = shuffle_in_order(train_images, train_labels)

    print('train: ', train_images.shape, train_labels.shape)
    print('test: ', test_images.shape, test_labels.shape)