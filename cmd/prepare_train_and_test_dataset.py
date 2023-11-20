import os
import gc
import sys
import numpy as np

def save_and_delete_data(path, filename, data):
    np.save(os.path.join(path, filename), data)
    del data
    gc.collect()

def separete_classes(train_images, train_labels):
    # Aqui vamos usar as máscaras diretamente nos rótulos
    mask_covid = train_labels == COVID
    mask_normal = train_labels == NORMAL
    mask_pneumonia = train_labels == PNEUMONIA
    mask_other_findings = train_labels == OTHER_FINDING

    # Agora, usamos as máscaras para selecionar as imagens correspondentes
    covid_images = train_images[mask_covid]
    normal_images = train_images[mask_normal]
    pneumonia_images = train_images[mask_pneumonia]
    other_findings_images = train_images[mask_other_findings]

    covid_labels = train_labels[mask_covid]
    normal_labels = train_labels[mask_normal]
    pneumonia_labels = train_labels[mask_pneumonia]
    other_findings_labels = train_labels[mask_other_findings]

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
# (train_infiltration_images, train_infiltration_labels), (test_infiltration_images, test_infiltration_labels) = load_infiltration_dataset()

# separar em classes
(covid_images, covid_labels), (normal_images, normal_labels), (pneumonia_images, pneumonia_labels), (other_findings_images, other_findings_labels) = separete_classes(train_images, train_labels)

del train_images
del train_labels

gc.collect()

train_images_subset = np.concatenate((covid_images[:num_train_samples_per_class], normal_images[:num_train_samples_per_class], pneumonia_images[:num_train_samples_per_class], other_findings_images[:num_train_samples_per_class]), axis=0)
train_labels_subset = np.concatenate((covid_labels[:num_train_samples_per_class], normal_labels[:num_train_samples_per_class], pneumonia_labels[:num_train_samples_per_class], other_findings_labels[:num_train_samples_per_class]), axis=0)

print('train_images.shape:', train_images_subset.shape, 'train_labels.shape: ', train_labels_subset.shape)

(train_images_subset, train_labels_subset) = shuffle_in_order(train_images_subset, train_labels_subset)

# Salve e exclua as fatias de treino
save_and_delete_data(PATH_ROOT, 'datasets/train/train_images.npy', train_images_subset)
save_and_delete_data(PATH_ROOT, 'datasets/train/train_labels.npy', train_labels_subset)

retrain_images_subset = np.concatenate((covid_images[num_train_samples_per_class:], normal_images[num_train_samples_per_class:], pneumonia_images[num_train_samples_per_class:], other_findings_images[num_train_samples_per_class:]), axis=0)
retrain_labels_subset = np.concatenate((covid_labels[num_train_samples_per_class:], normal_labels[num_train_samples_per_class:], pneumonia_labels[num_train_samples_per_class:], other_findings_labels[num_train_samples_per_class:]), axis=0)

print('retrain_images.shape:', retrain_images_subset.shape, 'retrain_labels.shape: ', retrain_labels_subset.shape)

(retrain_images, retrain_labels) = shuffle_in_order(retrain_images_subset, retrain_labels_subset)

# Salve e exclua as fatias de re-treino
save_and_delete_data(PATH_ROOT, 'datasets/train/retrain_images.npy', retrain_images_subset)
save_and_delete_data(PATH_ROOT, 'datasets/train/retrain_labels.npy', retrain_labels_subset)

print('test_images.shape:', test_images.shape, 'test_labels.shape: ', test_labels.shape)

(test_images, test_labels) = shuffle_in_order(test_images, test_labels)

# Salve e exclua as imagens de teste
save_and_delete_data(PATH_ROOT, 'datasets/test/test_images.npy', test_images)
save_and_delete_data(PATH_ROOT, 'datasets/test/test_labels.npy', test_labels)

print('======================================================')
print("Datasets de treino e validação preparados.")
print('======================================================')
