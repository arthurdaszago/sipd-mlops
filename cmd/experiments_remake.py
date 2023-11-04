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

num_samples = sys.argv[1]

from src.utils.detect_concept_drift import detect_experiment_remake_concept_drift

# ================================================

percents_of_unknown_samples = [5.0, 10.0, 15.0, 20.0]

# Carrega e compila o modelo
model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    conf_matrix = confusion_matrix(experiment_labels, predicted_labels)

    # Escrevendo no arquivo JSON
    stats = {
        "Accuracy": round(accuracy_score(experiment_labels, predicted_labels), 5),
        "Recall": round(recall_score(experiment_labels, predicted_labels, average='macro'), 5),
        "Precision": round(precision_score(experiment_labels, predicted_labels, average='macro'), 5),
        "F1-Score": round(f1_score(experiment_labels, predicted_labels, average='macro'), 5),
    }

    # Escrevendo no arquivo JSON
    with open(os.path.join(EXPERIMENT_STATS_PATH, f'retrained_stats_{percentage}.json'), 'w') as f:
        json.dump(stats, f)

    # Plotando a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(EXPERIMENT_STATS_PATH, f'confusion_matrix_{percentage}.png'))

    has_concept_drift = detect_experiment_remake_concept_drift(stats=stats)

    if has_concept_drift:
        parameters = { 'num_samples': num_samples }
        mlflow.run('.', entry_point='retrain_model', parameters=parameters)