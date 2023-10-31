
import os
import json
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
TEST_STATS_PATH = os.getenv('TEST_STATS_PATH')
TEST_DATASET_PATH = os.getenv('TEST_DATASET_PATH')

# ================================================

test_images = np.load(os.path.join(TEST_DATASET_PATH, 'test_images.npy'))
test_labels = np.load(os.path.join(TEST_DATASET_PATH, 'test_labels.npy'))

model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))

# Testando o modelo
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(test_labels, predicted_labels)

print('conf_matrix: ', conf_matrix)

# Calculando acurácia
accuracy = np.trace(conf_matrix) / float(np.sum(conf_matrix))

# Micro-average calculation
tp_total = np.sum(np.diag(conf_matrix))  # Sum of diagonal elements gives total true positives
fp_total = np.sum(np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))  # Sum of columns minus diagonal
fn_total = np.sum(np.sum(conf_matrix, axis=1) - np.diag(conf_matrix))  # Sum of rows minus diagonal
tn_total = np.sum(conf_matrix) - (tp_total + fp_total + fn_total)

print('tp_total: ', tp_total)
print('fp_total: ', fp_total)
print('fn_total: ', fn_total)
print('tn_total: ', tn_total)

# Para multiclasse, vamos calcular TPR e TNR para cada classe e armazená-los em listas
tpr_micro = tp_total / (tp_total + fn_total)
tnr_micro = tn_total / (tn_total + fp_total)

# Escrevendo no arquivo JSON
stats = {
    'TPR': tpr_micro,
    'TNR': tnr_micro,
    'accuracy': accuracy,
    'confusion_matrix': {
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'tn': tn_total,
        'list': conf_matrix.tolist(),
    },
}

# Escrevendo no arquivo JSON
with open(os.path.join(TEST_STATS_PATH, 'matrix_confusion.json'), 'w') as f:
    json.dump(stats, f)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(TEST_STATS_PATH, 'confusion_matrix.png'))