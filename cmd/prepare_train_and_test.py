import numpy as np
import tensorflow as tf
import os

PATH_ROOT = None
if os.getenv('PATH_ROOT') is not None:
    PATH_ROOT = os.getenv('PATH_ROOT')
else:
    PATH_ROOT = '/home/arthur/Documents/ifc/tc/code/sipd-mlops'

# Carregar o dataset CIFAR-10
(train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()

# Definindo os índices das classes no CIFAR-10
AVIÃO, AUTOMOBILE, PÁSSARO, GATO, VEADO, CACHORRO, SAPO, CAVALO, NAVIO, CAMINHÃO = range(10)

# Classes de interesse
individual_classes = [AVIÃO, CAMINHÃO, NAVIO]
other_classes = [PÁSSARO, GATO, VEADO, CACHORRO, CAVALO]
excluded_class = [AUTOMOBILE]

# Preparando as imagens e rótulos
mask_individual = np.isin(train_labels, individual_classes)
mask_others = np.isin(train_labels, other_classes)

images_individual = train_images[mask_individual.squeeze()]
labels_individual = train_labels[mask_individual.squeeze()]

images_others = train_images[mask_others.squeeze()]
labels_others = np.full(shape=images_others.shape[0], fill_value=3).reshape(-1, 1)  # Label 3 para "outros"

# Combinando as imagens e rótulos
train_images_final = np.vstack([images_individual, images_others])
train_labels_final = np.vstack([labels_individual, labels_others])

# Embaralhando e separando 10% para validação
p = np.random.permutation(train_images_final.shape[0])
train_images_final = train_images_final[p]
train_labels_final = train_labels_final[p]

test_images = train_images_final[:4000]
test_labels = train_labels_final[:4000]

train_images = train_images_final[4000:]
train_labels = train_labels_final[4000:]

print('train_labels: ', train_labels)

# Salvando os datasets
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_images.npy'), train_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_labels.npy'), train_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_images.npy'), test_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_labels.npy'), test_labels)

print("Datasets de treino e validação preparados.")