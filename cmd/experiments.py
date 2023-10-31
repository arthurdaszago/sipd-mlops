import os
import json
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
EXPERIMENT_STATS_PATH = os.getenv('EXPERIMENT_STATS_PATH')
EXPERIMENTS_DATASET_PATH = os.getenv('EXPERIMENTS_DATASET_PATH')

# ================================================

percents_of_unknown_samples = [5, 10, 20, 30]

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
    conf_matrix = confusion_matrix(experiment_labels, predicted_labels)

    # Calculando acurácia
    accuracy = np.trace(conf_matrix) / float(np.sum(conf_matrix))

    # Micro-average calculation
    tp_total = np.sum(np.diag(conf_matrix))  # Sum of diagonal elements gives total true positives
    fp_total = np.sum(np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))  # Sum of columns minus diagonal
    fn_total = np.sum(np.sum(conf_matrix, axis=1) - np.diag(conf_matrix))  # Sum of rows minus diagonal
    tn_total = np.sum(conf_matrix) - (tp_total + fp_total + fn_total)

    # Para multiclasse, vamos calcular TPR e TNR para cada classe e armazená-los em listas
    tpr_micro = tp_total / (tp_total + fn_total)
    tnr_micro = tn_total / (tn_total + fp_total)

    # Escrevendo no arquivo JSON
    stats = {
        'TPR': int(tpr_micro),
        'TNR': int(tnr_micro),
        'accuracy': accuracy,
        'confusion_matrix': {
            'tp': int(tp_total),
            'fp': int(fp_total),
            'fn': int(fn_total),
            'tn': int(tn_total),
            'list': conf_matrix.tolist(),
        },
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
