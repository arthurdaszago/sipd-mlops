
import os
import json
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score

# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
TEST_STATS_PATH = os.getenv('TEST_STATS_PATH')
TEST_DATASET_PATH = os.getenv('TEST_DATASET_PATH')

# ================================================

# Definindo os índices das classes no dataset
COVID, NORMAL, PNEUMONIA, OTHER_FINDINGS = range(4)

# Classes de interesse
unknown_class = [OTHER_FINDINGS]
known_classes = [COVID, NORMAL, PNEUMONIA]

test_images = np.load(os.path.join(TEST_DATASET_PATH, 'test_images.npy'))
test_labels = np.load(os.path.join(TEST_DATASET_PATH, 'test_labels.npy'))

model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))

# Testando o modelo
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

print('test_labels.shape: ', test_labels.shape)
print('test_labels.shape: ', predicted_labels.shape)

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(test_labels, predicted_labels, labels=known_classes)

# Escrevendo no arquivo JSON
stats = {
    "Accuracy": round(accuracy_score(test_labels, predicted_labels), 5),
    "Recall": round(recall_score(test_labels, predicted_labels, average='macro'), 5),
    "Precision": round(precision_score(test_labels, predicted_labels, average='macro'), 5),
    "F1-Score": round(f1_score(test_labels, predicted_labels, average='macro'), 5),
}

# Escrevendo no arquivo JSON
with open(os.path.join(TEST_STATS_PATH, 'stats.json'), 'w') as f:
    json.dump(stats, f)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(TEST_STATS_PATH, 'confusion_matrix.png'))