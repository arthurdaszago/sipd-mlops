import os
import sys
import numpy as np


def separete_classes(train_images, train_labels):
    # isin
    mask_covid_images = np.isin(train_images, [COVID])
    mask_covid_labels = np.isin(train_labels, [COVID])
    mask_normal_images = np.isin(train_images, [NORMAL])
    mask_normal_labels = np.isin(train_labels, [NORMAL])
    mask_pneumonia_images = np.isin(train_images, [PNEUMONIA])
    mask_pneumonia_labels = np.isin(train_labels, [PNEUMONIA])
    mask_other_findings_images = np.isin(train_images, [OTHER_FINDING])
    mask_other_findings_labels = np.isin(train_labels, [OTHER_FINDING])

    # images and labels by class
    covid_images = train_images[mask_covid_images.squeeze()]
    covid_labels = train_labels[mask_covid_labels.squeeze()]
    normal_images = train_images[mask_normal_images.squeeze()]
    normal_labels = train_labels[mask_normal_labels.squeeze()]
    pneumonia_images = train_images[mask_pneumonia_images.squeeze()]
    pneumonia_labels = train_labels[mask_pneumonia_labels.squeeze()]
    other_findings_images = train_images[mask_other_findings_images.squeeze()]
    other_findings_labels = train_labels[mask_other_findings_labels.squeeze()]

    return (covid_images, covid_labels), (normal_images, normal_labels), (pneumonia_images, pneumonia_labels), (other_findings_images, other_findings_labels)

# ================================================

num_train_samples_per_class = 8000
num_total_samples_per_class = 10000

COVID, NORMAL, PNEUMONIA, OTHER_FINDING = range(4)

PATH_ROOT = os.getenv('PATH_ROOT')
sys.path.append(PATH_ROOT)

from src.utils.load_dataset import load_dataset, shuffle_in_order
from src.utils.load_infiltration_dataset import load_dataset as load_infiltration_dataset

# ================================================

print('======================================================')
print("Iniciando preparamento de datasets de treino e validação.")
print('======================================================')

# Carregar o dataset 
(train_images, train_labels), (test_images, test_labels) = load_dataset()
(train_infiltration_images, train_infiltration_labels), (test_infiltration_images, test_infiltration_labels) = load_infiltration_dataset()

# separar em classes
(covid_images, covid_labels), (normal_images, normal_labels), (pneumonia_images, pneumonia_labels), (other_findings_images, other_findings_labels) = separete_classes(train_images, train_labels)

train_images = np.concatenate((covid_images[:num_train_samples_per_class], normal_images[:num_train_samples_per_class], pneumonia_images[:num_train_samples_per_class], other_findings_images[:num_train_samples_per_class]))
train_labels = np.concatenate((covid_labels[:num_train_samples_per_class], normal_labels[:num_train_samples_per_class], pneumonia_labels[:num_train_samples_per_class], other_findings_labels[:num_train_samples_per_class]))

print('covid_images[num_train_samples_per_class:].shape: ', covid_images[num_train_samples_per_class:].shape)

retrain_images = np.concatenate((covid_images[num_train_samples_per_class:num_total_samples_per_class], normal_images[num_train_samples_per_class:num_total_samples_per_class], pneumonia_images[num_train_samples_per_class:num_total_samples_per_class], other_findings_images[num_train_samples_per_class:num_total_samples_per_class]))
retrain_labels = np.concatenate((covid_labels[num_train_samples_per_class:], normal_labels[num_train_samples_per_class:], pneumonia_labels[num_train_samples_per_class:], other_findings_labels[num_train_samples_per_class:]))

(test_images, test_labels) = shuffle_in_order(test_images, test_labels)
(train_images, train_labels) = shuffle_in_order(train_images, train_labels)
# (retrain_images, retrain_labels) = shuffle_in_order(retrain_images, retrain_labels)

print('test_images.shape:', test_images.shape, 'test_labels.shape: ', test_labels.shape)
print('train_images.shape:', train_images.shape, 'train_labels.shape: ', train_labels.shape)
# print('retrain_images.shape:', retrain_images.shape, 'retrain_labels.shape: ', retrain_labels.shape)

# Salvando os datasets
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_images.npy'), test_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_labels.npy'), test_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_images.npy'), train_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_labels.npy'), train_labels)

np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'retrain_covid_images.npy'), retrain_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'retrain_covid_labels.npy'), retrain_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'retrain_normal_images.npy'), retrain_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'retrain_normal_labels.npy'), retrain_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'retrain_pneumonia_images.npy'), retrain_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'retrain_pneumonia_labels.npy'), retrain_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'retrain_other_findings_images.npy'), retrain_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'retrain_other_findings_labels.npy'), retrain_labels)

np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_infiltration_images.npy'), test_infiltration_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_infiltration_labels.npy'), test_infiltration_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_infiltration_images.npy'), train_infiltration_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_infiltration_labels.npy'), train_infiltration_labels)

print('======================================================')
print("Datasets de treino e validação preparados.")
print('======================================================')
