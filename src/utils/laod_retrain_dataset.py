import os
import numpy as np

# ==================================================
TRAIN_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')
# ==================================================

COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS = range(4)


def add_samples_in_others(other_findings, infiltrations, num_samples):
    (infiltrations_images, infiltrations_labels) = infiltrations
    (other_findings_images, other_findings_labels) = other_findings

    new_other_findings_images = np.concatenate((other_findings_images[num_samples:], infiltrations_images[:num_samples]), axis=0)
    new_other_findings_labels = np.concatenate((other_findings_labels[num_samples:], infiltrations_labels[:num_samples]), axis=0)

    return new_other_findings_images, new_other_findings_labels

def load_retrain_dataset(tax_samples: float):
    num_samples = tax_samples * 2000

    retrain_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'retrain_images.npy'))
    retrain_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'retrain_labels.npy'))

    covid_images = retrain_images[(retrain_labels == COVID)]     
    covid_labels = retrain_labels[(retrain_labels == COVID)]    

    normal_images = retrain_images[(retrain_labels == NORMAL)]     
    normal_labels = retrain_labels[(retrain_labels == NORMAL)]    

    pneumonia_images = retrain_images[(retrain_labels == PNEUMONIA)]     
    pneumonia_labels = retrain_labels[(retrain_labels == PNEUMONIA)]    

    other_findings_images = retrain_images[(retrain_labels == OTHER_FINDINGS)]     
    other_findings_labels = retrain_labels[(retrain_labels == OTHER_FINDINGS)]     

    infiltrations_images = np.load(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_infiltration_images.npy'))
    infiltrations_labels = np.load(os.path.join(TRAIN_DATASET_PATH, 'original', 'train_infiltration_labels.npy'))

    other_findings_images, other_findings_labels =  add_samples_in_others(other_findings=(other_findings_images, other_findings_labels), infiltrations=(infiltrations_images, infiltrations_labels), num_samples=int(num_samples))

    return (covid_images, covid_labels), (normal_images, normal_labels), (pneumonia_images, pneumonia_labels), (other_findings_images, other_findings_labels) 


