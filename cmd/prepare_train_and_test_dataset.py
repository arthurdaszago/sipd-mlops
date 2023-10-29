import os
import sys
import numpy as np

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
sys.path.append(PATH_ROOT)

from src.utils.load_dataset import load_dataset, shuffle_in_order

# ================================================

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

mask_covid = np.isin(train_labels, [COVID])
mask_normal = np.isin(train_labels, [NORMAL])
mask_pneumonia = np.isin(train_labels, [PNEUMONIA])

covid_images = train_images[mask_covid.squeeze()]
normal_images = train_images[mask_normal.squeeze()]
pneumonia_images = train_images[mask_pneumonia.squeeze()]

covid_labels = train_labels[mask_covid.squeeze()]
normal_labels = train_labels[mask_normal.squeeze()]
pneumonia_labels = train_labels[mask_pneumonia.squeeze()]

# Separando 1500 para validação
validation_images = np.vstack([ covid_images[:500], normal_images[:500], pneumonia_images[:500]])
validation_labels = np.hstack([ covid_labels[:500], normal_labels[:500], pneumonia_labels[:500]])

(validation_images, validation_labels) = shuffle_in_order(validation_images, validation_labels)

# validation_images = known_images[:1500]
# validation_labels = known_labels[:1500]

print('validation_images shape: ', validation_images.shape)
print('validation_labels shape: ', validation_labels.shape)

train_images = np.vstack([ covid_images[500:], normal_images[500:], pneumonia_images[500:]])
train_labels = np.hstack([ covid_labels[500:], normal_labels[500:], pneumonia_labels[500:]])

(train_images, train_labels) = shuffle_in_order(train_images, train_labels)

# train_images = known_images[1500:]
# train_labels = known_labels[1500:]

print('train_images shape: ', train_images.shape)
print('train_labels shape: ', train_labels.shape)

# Salvando os datasets
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_images.npy'), train_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_labels.npy'), train_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_other_findings_images.npy'), unknown_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_other_findings_labels.npy'), unknown_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'validation_images.npy'), validation_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'validation_labels.npy'), validation_labels)

print("Datasets de treino e validação preparados.")