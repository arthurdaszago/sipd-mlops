import os
import sys
import numpy as np

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
sys.path.append(PATH_ROOT)

from src.utils.load_dataset import load_dataset, shuffle_in_order

# ================================================

print('======================================================')
print("Iniciando preparamento de datasets de treino e validação.")
print('======================================================')

# Carregar o dataset 
(train_images, train_labels), (test_images, test_labels) = load_dataset()

# Definindo os índices das classes no dataset
COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS = range(4)

# Classes de interesse
unknown_class = [OTHER_FINDINGS]
known_classes = [COVID, NORMAL, PNEUMONIA]

# Preparando as imagens e rótulos
mask_known = np.isin(train_labels, known_classes)
mask_unknown = np.isin(train_labels, unknown_class)

known_images = train_images[mask_known.squeeze()]
known_labels = train_labels[mask_known.squeeze()]

print('known_images shape: ', known_images.shape)
print('known_labels shape: ', known_labels.shape)

unknown_images = train_images[mask_unknown.squeeze()]
unknown_labels = train_labels[mask_unknown.squeeze()]

print('unknown_images shape: ', unknown_images.shape)
print('unknown_labels shape: ', unknown_labels.shape)

mask_covid = np.isin(test_labels, [COVID])
mask_normal = np.isin(test_labels, [NORMAL])
mask_pneumonia = np.isin(test_labels, [PNEUMONIA])

covid_images = test_images[mask_covid.squeeze()]
normal_images = test_images[mask_normal.squeeze()]
pneumonia_images = test_images[mask_pneumonia.squeeze()]

covid_labels = test_labels[mask_covid.squeeze()]
normal_labels = test_labels[mask_normal.squeeze()]
pneumonia_labels = test_labels[mask_pneumonia.squeeze()]

# Separando 1500 para validação
test_images = np.vstack([ covid_images, normal_images, pneumonia_images])
test_labels = np.hstack([ covid_labels, normal_labels, pneumonia_labels])

print('test_images.shape: ', test_images.shape)
print('test_labels.shape: ', test_labels.shape)

train_images = np.vstack([ known_images ])
train_labels = np.hstack([ known_labels ])

print('train_images shape: ', train_images.shape)
print('train_labels shape: ', train_labels.shape)

(test_images, test_labels) = shuffle_in_order(test_images, test_labels)
(train_images, train_labels) = shuffle_in_order(train_images, train_labels)

# Salvando os datasets
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_images.npy'), test_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_labels.npy'), test_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_images.npy'), train_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_labels.npy'), train_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_other_findings_images.npy'), unknown_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_other_findings_labels.npy'), unknown_labels)

print('======================================================')
print("Datasets de treino e validação preparados.")
print('======================================================')