import os
import sys
import json
import mlflow
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
EXPERIMENT_STATS_PATH = os.getenv('EXPERIMENT_STATS_PATH')
EXPERIMENTS_DATASET_PATH = os.getenv('EXPERIMENTS_DATASET_PATH')

sys.path.append(PATH_ROOT)

from src.utils.detect_concept_drift import detect_experimet_concept_drift 

# ================================================

# Definindo os índices das classes no dataset
COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS = range(4)

# Classes de interesse
unknown_class = [OTHER_FINDINGS]
known_classes = [COVID, NORMAL, PNEUMONIA]

percents_of_unknown_samples = [5.0, 10.0, 15.0, 20.0]

model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))

for percentage in percents_of_unknown_samples:
    experiment_images_path = os.path.join(EXPERIMENTS_DATASET_PATH, f'test_data_{percentage}_other_findings.npy')
    experiment_labels_path = os.path.join(EXPERIMENTS_DATASET_PATH, f'test_label_{percentage}_other_findings.npy')

    experiment_images = np.load(experiment_images_path)
    experiment_labels = np.load(experiment_labels_path)

    print('experiment_images: ', experiment_images.shape)
    print('experiment_labels: ', experiment_labels.shape)

    # Testando o modelo
    predictions = model.predict(experiment_images)
    predicted_labels = np.argmax(predictions, axis=1)

    print('predicted_labels: ', predicted_labels.shape)
    print('experiment_labels: ', experiment_labels.shape)

    # Calculando a matriz de confusão
    conf_matrix = confusion_matrix(experiment_labels, predicted_labels, labels=known_classes)

    # Escrevendo no arquivo JSON
    stats = {
        "Accuracy": round(accuracy_score(experiment_labels, predicted_labels), 5),
        "Recall": round(recall_score(experiment_labels, predicted_labels, average='macro'), 5),
        "Precision": round(precision_score(experiment_labels, predicted_labels, average='macro'), 5),
        "F1-Score": round(f1_score(experiment_labels, predicted_labels, average='macro'), 5),
    }

    # Escrevendo no arquivo JSON
    with open(os.path.join(EXPERIMENT_STATS_PATH, f'stats_{percentage}.json'), 'w') as f:
        print('Salvando.......................')
        json.dump(stats, f)
        print('Salvou.......................')

    # Plotando a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(EXPERIMENT_STATS_PATH, f'confusion_matrix_{percentage}.png'))

    has_concept_drift = detect_experimet_concept_drift(stats=stats)

    if has_concept_drift:
        parameters = { 'num_train_samples': None }
        mlflow.run('.', 'train_remake_model', parameters=parameters)