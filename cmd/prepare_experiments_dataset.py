import os
import sys
import numpy as np

# ================================================
COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS = range(4)

# num of total samples
total_samples = 2000
# percent of unknown samples to add in experiment
percentages = [0.05, 0.1, 0.15, 0.2]

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
sys.path.append(PATH_ROOT)

from src.utils.load_dataset import load_dataset, shuffle_in_order
from src.utils.load_infiltration_dataset import load_dataset as load_infiltration_dataset

# ================================================

print('======================================================')
print("Iniciando preparação de datasets de experimentos.")
print('======================================================')

_, (test_images, test_labels) = load_dataset()
_, (test_infiltration_images, test_infiltration_labels) = load_infiltration_dataset()

print('test_images.shape: ', test_images.shape, ', test_labels.shape: ', test_labels.shape)
print('test_infiltration_images.shape: ', test_infiltration_images.shape, ', test_infiltration_labels.shape: ', test_infiltration_labels.shape)

mask_known_images = np.isin(test_images, [COVID, NORMAL, PNEUMONIA])
mask_known_labels = np.isin(test_labels, [COVID, NORMAL, PNEUMONIA])

known_test_labels = test_labels[(test_labels == COVID) | (test_labels ==  NORMAL) | (test_labels ==  PNEUMONIA)]
known_test_images = test_images[(test_labels == COVID) | (test_labels ==  NORMAL) | (test_labels ==  PNEUMONIA)]

print('known_test_images.shape: ', known_test_images.shape)
print('known_test_labels.shape: ', known_test_labels.shape)

other_finding_test_images = test_images[(test_labels == OTHER_FINDINGS)]
other_finding_test_labels = test_labels[(test_labels == OTHER_FINDINGS)]

print('other_finding_test_images.shape: ', other_finding_test_images.shape)
print('other_finding_test_labels.shape: ', other_finding_test_labels.shape)

for percentage in percentages:
    num_samples_by_percent = int(percentage * (total_samples // 4))
    print('num_samples_by_percent: ', num_samples_by_percent)

    other_finding_images = other_finding_test_images[num_samples_by_percent:500]
    other_finding_labels = other_finding_test_labels[num_samples_by_percent:500]

    infiltration_images = test_infiltration_images[:num_samples_by_percent]
    infiltration_labels = test_infiltration_labels[:num_samples_by_percent]

    print('known_test_images[:1500].shape: ', known_test_images.shape[:1500], ', other_finding_images.shape:', other_finding_images.shape, ', infiltration_images.shape:', infiltration_images.shape)
    print('known_test_labels.shape: ', known_test_labels.shape, ', other_finding_labels.shape:', other_finding_labels.shape, ', infiltration_labels.shape:', infiltration_labels.shape)

    # Salvando
    final_images = np.concatenate((known_test_images[:1500], other_finding_images, infiltration_images), axis=0)
    final_labels = np.concatenate((known_test_labels[:1500], other_finding_labels, infiltration_labels), axis=0)

    print('final_images: ', final_images.shape)
    print('final_labels: ', final_labels.shape)
    
    final_images, final_labels = shuffle_in_order(images=final_images, labels=final_labels)

    # Contando e exibindo o número de amostras para cada classe
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    for ulabel, count in zip(unique_labels, counts):
        class_name = [ 'COVID', 'NORMAL', 'PNEUMONIA', 'OTHER_FINDINGS' ][ulabel]
        print(f"For {int(percentage*100)}% frogs: Number of samples for {class_name}: {count}")

    np.save(os.path.join(PATH_ROOT, 'datasets', 'experiments', f'experiment_images_{percentage*100}_infiltration.npy'), final_images)
    np.save(os.path.join(PATH_ROOT, 'datasets', 'experiments', f'experiment_labels_{percentage*100}_infiltration.npy'), final_labels)

print('======================================================')
print("tERMINADO preparação de datasets de experimentos.")
print('======================================================')