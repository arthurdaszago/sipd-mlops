
import os
import sys
import json
import mlflow
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, recall_score, accuracy_score, precision_score, f1_score

# ================================================

classes = ['COVID', 'Normal', 'Pneumonia', 'Outras Doenças']

PATH_ROOT = os.getenv('PATH_ROOT')
TEST_STATS_PATH = os.getenv('TEST_STATS_PATH')
TEST_DATASET_PATH = os.getenv('TEST_DATASET_PATH')

tax_samples = float(sys.argv[1])

# ================================================

test_images = np.load(os.path.join(TEST_DATASET_PATH, 'test_images.npy'))
test_labels = np.load(os.path.join(TEST_DATASET_PATH, 'test_labels.npy'))

model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model_retrained.h5'))

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Testando o modelo
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(test_labels, predicted_labels)
m_conf_matrix = multilabel_confusion_matrix(test_labels, predicted_labels)

# Escrevendo no arquivo JSON
stats = {
    "confusion-matrix": conf_matrix.tolist(),
    "multilabel-confusion-marix": m_conf_matrix.tolist(),
    "Accuracy": round(accuracy_score(test_labels, predicted_labels), 5),
    "Recall": round(recall_score(test_labels, predicted_labels, average='macro'), 5),
    "Precision": round(precision_score(test_labels, predicted_labels, average='macro'), 5),
    "F1-Score": round(f1_score(test_labels, predicted_labels, average='macro'), 5),
}

# Escrevendo no arquivo JSON
with open(os.path.join(TEST_STATS_PATH, 'retest_stats.json'), 'w') as f:
    json.dump(stats, f)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de confusão')
plt.savefig(os.path.join(TEST_STATS_PATH, 'retest_confusion_matrix.png'))

parameters = { 'tax_samples': tax_samples }
mlflow.run('.', entry_point='remake_experiments', parameters=parameters)