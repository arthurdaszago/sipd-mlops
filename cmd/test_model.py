import os
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
if os.getenv('TRAIN_DATASET_PATH') is not None:
    TEST_DATASET_PATH = os.getenv('TRAIN_DATASET_PATH')
else:
    TEST_DATASET_PATH = '/home/arthur/Documents/ifc/tc/code/sipd-mlops/datasets/test'

test_images = np.load(os.path.join(TEST_DATASET_PATH, 'train_images.npy'))
test_labels = np.load(os.path.join(TEST_DATASET_PATH, 'train_labels.npy'))

model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))

# Testando o modelo
# Calculando a matriz de confusão
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()