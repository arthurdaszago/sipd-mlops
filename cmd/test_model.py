import os
import json
import numpy as np
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


PATH_ROOT = None
if os.getenv('PATH_ROOT') is not None:
    PATH_ROOT = os.getenv('PATH_ROOT')
else:
    PATH_ROOT = '/home/arthur/Documents/ifc/tc/code/sipd-mlops'

TEST_DATASET_PATH = None
if os.getenv('TEST_DATASET_PATH') is not None:
    TEST_DATASET_PATH = os.getenv('TEST_DATASET_PATH')
else:
    TEST_DATASET_PATH = '/home/arthur/Documents/ifc/tc/code/sipd-mlops/datasets/test'

TEST_STATS_PATH = None
if os.getenv('TEST_STATS_PATH') is not None:
    TEST_STATS_PATH = os.getenv('TEST_STATS_PATH')
else:
    TEST_STATS_PATH = '/home/arthur/Documents/ifc/tc/code/sipd-mlops/stats/test'

test_images = np.load(os.path.join(TEST_DATASET_PATH, 'test_images.npy'))
test_labels = np.load(os.path.join(TEST_DATASET_PATH, 'test_labels.npy'))

model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))

# Testando o modelo
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Calculando acurácia
accuracy = np.trace(conf_matrix) / float(np.sum(conf_matrix))

# Para multiclasse, vamos calcular TPR e TNR para cada classe e armazená-los em listas
tpr_list = []
tnr_list = []

for i in range(conf_matrix.shape[0]):
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
with open(os.path.join(TEST_STATS_PATH, 'matrix_confusion.json'), 'w') as f:
    json.dump(stats, f)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(TEST_STATS_PATH, 'confusion_matrix.png'))