import os
import numpy as np

from PIL import Image

PATH_ROOT = os.getenv('PATH_ROOT')
DATASET_PATH = os.getenv('DATASET_PATH')
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
     train_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_images.npy'))
     train_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_labels.npy'))
    # load default test dataset
     test_images = np.load(os.path.join(TEST_DATASET_PATH, 'original', 'test_images.npy'))
     test_labels = np.load(os.path.join(TEST_DATASET_PATH, 'original', 'test_labels.npy'))

     return (train_images, train_labels), (test_images, test_labels)

def save_dataset(train, test):
    (test_images, test_labels) = test
    (train_images, train_labels) = train

    # save default train dataset
    np.save(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_images.npy'), train_images)
    np.save(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_labels.npy'), train_labels)
    # save default test dataset
    np.save(os.path.join(TEST_DATASET_PATH, 'original', 'test_images.npy'), test_images)
    np.save(os.path.join(TEST_DATASET_PATH, 'original', 'test_labels.npy'), test_labels)

def already_exists():
    train_images_exists = os.path.exists(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_images.npy'))
    train_labels_exists = os.path.exists(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_labels.npy'))
    test_images_exists = os.path.exists(os.path.join(TEST_DATASET_PATH, 'original', 'test_images.npy'))
    test_labels_exists = os.path.exists(os.path.join(TEST_DATASET_PATH, 'original', 'test_labels.npy'))

    if train_images_exists and train_labels_exists and test_images_exists and test_labels_exists:
        return True

    return False


def load_train_dataset():
    train_dataset_path = os.path.join(DATASET_PATH, 'train')

    train_folders = os.listdir(train_dataset_path)

    images = []
    labels = []

    for train_folder in train_folders:
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
                case 'other_findings':
                    labels.append(3)

    return np.asarray(images), np.asarray(labels)


def load_test_dataset():
    test_dataset_path = os.path.join(DATASET_PATH, 'test')

    test_folders = os.listdir(test_dataset_path)

    images = []
    labels = []

    for test_folder in test_folders:
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
                case 'other_findings':
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