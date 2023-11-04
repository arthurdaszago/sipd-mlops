import os
import numpy as np

# ==================================================
TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')
# ==================================================

def load_retrain_dataset(num_samples: int):
    covid_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'retrain_covid_images.npy'))
    covid_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'retrain_covid_labels.npy'))

    normal_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'retrain_normal_images.npy'))
    normal_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'retrain_normal_labels.npy'))

    pneumonia_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'retrain_pneumonia_images.npy'))
    pneumonia_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'retrain_pneumonia_labels.npy'))

    other_findings_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_other_findings_images.npy'))
    other_findings_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_other_findings_labels.npy'))

    infiltrations_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_infiltration_images.npy'))
    infiltrations_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'train_infiltration_labels.npy'))

    other_findings_images, other_findings_labels =  add_samples_in_others(other_findings=(other_findings_images, other_findings_labels), infiltrations=(infiltrations_images, infiltrations_labels), num_samples=num_samples)

    return (covid_images[:num_samples], covid_labels[:num_samples]), (normal_images[:num_samples], normal_labels[:num_samples]), (pneumonia_images[:num_samples], pneumonia_labels[:num_samples]), (other_findings_images, other_findings_labels) 

def add_samples_in_others(other_findings, infiltrations, num_samples):
    (infiltrations_images, infiltrations_labels) = infiltrations
    (other_findings_images, other_findings_labels) = other_findings

    new_other_findings_images = np.vstack((other_findings_images[num_samples:], infiltrations_images[:num_samples]))
    new_other_findings_labels = np.vstack((other_findings_labels[num_samples:], infiltrations_labels[:num_samples]))

    return new_other_findings_images, new_other_findings_labels

