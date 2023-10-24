import numpy as np
import tensorflow as tf
import os

PATH_ROOT = None
if os.getenv('PATH_ROOT') is not None:
    PATH_ROOT = os.getenv('PATH_ROOT')
else:
    PATH_ROOT = '/home/arthur/Documents/ifc/tc/code/sipd-mlops'

_, (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Definindo os índices das classes no CIFAR-10
AVIÃO, AUTOMOBILE, PÁSSARO, GATO, VEADO, CACHORRO, SAPO, CAVALO, NAVIO, CAMINHÃO = range(10)

# Classes de interesse e excluídas
individual_classes = [AVIÃO, CAMINHÃO, NAVIO]
other_classes = [PÁSSARO, GATO, VEADO, CACHORRO, CAVALO]
excluded_class = [AUTOMOBILE]

percentages = [0.05, 0.1, 0.2, 0.5]
frog_samples = test_images[test_labels.squeeze() == SAPO]

for percentage in percentages:
    images_list = []
    labels_list = []

    # Adicionando classes individuais
    for label in individual_classes:
        mask = test_labels.squeeze() == label
        samples = test_images[mask][:1000]
        images_list.append(samples)
        labels_list.append(np.full((1000, 1), label))

    # Adicionando classe "outros"
    num_frog_samples = int(percentage * 1000)
    remaining_samples = 1000 - num_frog_samples
    num_other_classes = len(other_classes)
    samples_per_class = remaining_samples // num_other_classes

    other_samples = []
    for label in other_classes:
        mask = test_labels.squeeze() == label
        samples = test_images[mask][:samples_per_class]
        other_samples.append(samples)

    other_samples = np.vstack(other_samples)
    other_samples = np.vstack([other_samples, frog_samples[:num_frog_samples]])
    images_list.append(other_samples)
    labels_list.append(np.full((1000, 1), 3))  # Label 3 para "outros"

    # Salvando
    final_images = np.vstack(images_list)
    final_labels = np.vstack(labels_list)

    np.save(os.path.join(PATH_ROOT, 'datasets', 'experiments', f'test_data_{int(percentage*100)}_frog.npy'), final_images)
    np.save(os.path.join(PATH_ROOT, 'datasets', 'experiments', f'test_label_{int(percentage*100)}_frog.npy'), final_labels)

print("Datasets de experimentos preparados.")
