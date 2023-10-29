import os
import sys
import numpy as np

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
sys.path.append(PATH_ROOT)

from src.utils.load_dataset import load_dataset

# ================================================

_, (test_images, test_labels) = load_dataset()

# Definindo os índices das classes no dataset
COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS = range(4)

# Classes de interesse
unknown_class = [OTHER_FINDINGS]
known_classes = [COVID, NORMAL, PNEUMONIA]

# percent of unknown samples to add in experiment
percentages = [0.05, 0.15, 0.25, 0.35]

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
    images_list.append(other_finding_images_samples[:num_unknown_samples])
    labels_list.append(np.full((num_unknown_samples, 1), OTHER_FINDINGS))

    # Salvando
    final_images = np.vstack([images_list[0], images_list[1], images_list[2], images_list[3]])
    final_labels = np.vstack([labels_list[0], labels_list[1], labels_list[2], labels_list[3]])

    # Contando e exibindo o número de amostras para cada classe
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    for ulabel, count in zip(unique_labels, counts):
        class_name = [ 'COVID', 'NORMAL', 'PNEUMONIA', 'OTHER_FINDINGS' ][ulabel]
        print(f"For {int(percentage*100)}% frogs: Number of samples for {class_name}: {count}")

    np.save(os.path.join(PATH_ROOT, 'datasets', 'experiments', f'test_data_{int(percentage*100)}_other_findings.npy'), final_images)
    np.save(os.path.join(PATH_ROOT, 'datasets', 'experiments', f'test_label_{int(percentage*100)}_other_findings.npy'), final_labels)

print("Datasets de experimentos preparados.")
