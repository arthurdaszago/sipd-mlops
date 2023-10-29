import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
EXPERIMENT_STATS_PATH = os.getenv('EXPERIMENT_STATS_PATH')
EXPERIMENTS_DATASET_PATH = os.getenv('EXPERIMENTS_DATASET_PATH')

# ================================================

percents_of_unknown_samples = [5, 15, 25, 35]

model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))

for percentage in percents_of_unknown_samples:
    experiment_images_path = os.path.join(EXPERIMENTS_DATASET_PATH, f'test_data_{percentage}_other_findings.npy')
    experiment_labels_path = os.path.join(EXPERIMENTS_DATASET_PATH, f'test_label_{percentage}_other_findings.npy')

    experiment_images = np.load(experiment_images_path)
    experiment_labels = np.load(experiment_labels_path)

    # Testando o modelo
    predictions = model.predict(experiment_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculando a matriz de confusão
    conf_matrix = confusion_matrix(experiment_labels, predicted_labels)

    # Calculando acurácia
    accuracy = np.trace(conf_matrix) / float(np.sum(conf_matrix))

    # Para multiclasse, vamos calcular TPR e TNR para cada classe e armazená-los em listas
    tpr_list = []
    tnr_list = []

    for i in range(3):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - (tp + fp + fn)

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        tpr_list.append(tpr)
        tnr_list.append(tnr)

    # Escrevendo no arquivo JSON
    stats = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'TPR': tpr_list,
        'TNR': tnr_list
    }

    # Escrevendo no arquivo JSON
    with open(os.path.join(EXPERIMENT_STATS_PATH, f'matrix_confusion_{percentage}.json'), 'w') as f:
        json.dump(stats, f)

    # Plotando a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(EXPERIMENT_STATS_PATH, f'confusion_matrix_{percentage}.png'))
