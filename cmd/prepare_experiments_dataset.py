import os
import sys
import numpy as np

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
sys.path.append(PATH_ROOT)

from src.utils.load_dataset import load_dataset

# ================================================

print('======================================================')
print("Iniciando preparação de datasets de experimentos.")
print('======================================================')

_, (test_images, test_labels) = load_dataset()

# Definindo os índices das classes no dataset
COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS = range(4)

# Classes de interesse
unknown_class = [OTHER_FINDINGS]
known_classes = [COVID, NORMAL, PNEUMONIA]

# percent of unknown samples to add in experiment
percentages = [0.05, 0.1, 0.15, 0.2]

# num of total samples
total_samples = 1500

mask_unknown = np.isin(test_labels, unknown_class)

# all "other finding" samples
other_finding_images_samples = test_images[mask_unknown.squeeze()]
other_finding_labels_samples = test_labels[mask_unknown.squeeze()]

for percentage in percentages:
    samples_per_class = int((total_samples / len(known_classes)) - (percentage * total_samples / len(known_classes)))
    print('samples_per_class: ', samples_per_class)

    images_list = []
    labels_list = []

    # Adicionando classes individuais
    for label in known_classes:
        mask = test_labels.squeeze() == label
        samples = test_images[mask][:samples_per_class]
        images_list.append(samples)
        labels_list.append(np.full((samples_per_class, 1), label))

    # Adicionando classe desconhecida "others findings"
    num_unknown_samples = int(percentage * total_samples)

    # Determinar a quantidade de amostras em cada terço
    third_samples = num_unknown_samples // 3

    # Adicionar imagens e rótulos para o label 0
    images_list.append(other_finding_images_samples[:third_samples])
    labels_list.append(np.full((third_samples, 1), 0))

    print('other_finding_images_samples[:third_samples]: ', other_finding_images_samples[:third_samples].shape)

    # Adicionar imagens e rótulos para o label 1
    images_list.append(other_finding_images_samples[third_samples:2*third_samples])
    labels_list.append(np.full((third_samples, 1), 1))

    print('other_finding_images_samples[third_samples:2*third_samples]: ', other_finding_images_samples[third_samples:2*third_samples].shape)

    # Adicionar imagens e rótulos para o label 2
    images_list.append(other_finding_images_samples[2*third_samples:3*third_samples])
    labels_list.append(np.full((third_samples, 1), 2))

    print('other_finding_images_samples[2*third_samples:3*third_samples]: ', other_finding_images_samples[2*third_samples:3*third_samples].shape)

    print('other_finding_images_samples[:num_unknown_samples]: ', other_finding_images_samples[:num_unknown_samples].shape)

    # Salvando
    final_images = np.vstack(images_list)
    final_labels = np.vstack(labels_list)

    print('final_images: ', final_images.shape)    
    print('final_labels: ', final_labels.shape)    

    # Contando e exibindo o número de amostras para cada classe
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    for ulabel, count in zip(unique_labels, counts):
        class_name = [ 'COVID', 'NORMAL', 'PNEUMONIA', 'OTHER_FINDINGS' ][ulabel]
        print(f"For {int(percentage*100)}% frogs: Number of samples for {class_name}: {count}")

    np.save(os.path.join(PATH_ROOT, 'datasets', 'experiments', f'test_data_{percentage*100}_other_findings.npy'), final_images)
    np.save(os.path.join(PATH_ROOT, 'datasets', 'experiments', f'test_label_{percentage*100}_other_findings.npy'), final_labels)

print("Datasets de experimentos preparados.")
