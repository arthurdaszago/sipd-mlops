import os
import sys
import json
import mlflow
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, recall_score, accuracy_score, precision_score, f1_score

# ================================================

# Nomes das classes
classes = ['COVID', 'Normal', 'Pneumonia', 'Outras Doenças']

PATH_ROOT = os.getenv('PATH_ROOT')
EXPERIMENT_STATS_PATH = os.getenv('EXPERIMENT_STATS_PATH')
EXPERIMENTS_DATASET_PATH = os.getenv('EXPERIMENTS_DATASET_PATH')

sys.path.append(PATH_ROOT)

from src.utils.detect_concept_drift import detect_experimet_concept_drift 

# ================================================

# Definindo os índices das classes no dataset
COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS = range(4)

# Classes de interesse
known_classes = [COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS]

percents_of_unknown_samples = [10.0, 20.0, 30.0, 40.0]

pass

# Carrega e compila o modelo
model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

for percentage in percents_of_unknown_samples:
    experiment_images_path = os.path.join(EXPERIMENTS_DATASET_PATH, f'experiment_images_{percentage}_infiltration.npy')
    experiment_labels_path = os.path.join(EXPERIMENTS_DATASET_PATH, f'experiment_labels_{percentage}_infiltration.npy')

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
    m_conf_matrix = multilabel_confusion_matrix(experiment_labels, predicted_labels, labels=known_classes)

    # Escrevendo no arquivo JSON
    stats = {
        "confusion-matrix": conf_matrix.tolist(),
        "multilabel-confusion-marix": m_conf_matrix.tolist(),
        "Accuracy": round(accuracy_score(experiment_labels, predicted_labels), 5),
        "Recall": round(recall_score(experiment_labels, predicted_labels, average='macro'), 5),
        "Precision": round(precision_score(experiment_labels, predicted_labels, average='macro'), 5),
        "F1-Score": round(f1_score(experiment_labels, predicted_labels, average='macro'), 5),
    }

    # Escrevendo no arquivo JSON
    with open(os.path.join(EXPERIMENT_STATS_PATH, f'stats_{percentage}.json'), 'w') as f:
        json.dump(stats, f)

    # Plotando a matriz de confusão
    plt.rc('font', size=12)
    sns.set_context('talk')
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 17})
    plt.xlabel('Predito', fontsize=14)
    plt.ylabel('Verdadeiro', fontsize=14)
    plt.title(f'MC: {round(percentage / 4, 1)}% de amostras desconhecidas', fontsize=20)
    plt.savefig(os.path.join(EXPERIMENT_STATS_PATH, f'confusion_matrix_{percentage}.png'))

    has_concept_drift = detect_experimet_concept_drift(stats=stats)

    if has_concept_drift:
        mlflow.run('.', 'retrain_model', parameters={'tax_samples': round(percentage / 100, 1)})