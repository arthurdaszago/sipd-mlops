
import os
import json
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


from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix, recall_score, accuracy_score, precision_score, f1_score

COVID, NORMAL, PNEUMONIA, OTHER_FINDING = range(4)

def separete_classes(train_images, train_labels):
    # Aqui vamos usar as máscaras diretamente nos rótulos
    mask_covid = train_labels == COVID
    mask_normal = train_labels == NORMAL
    mask_pneumonia = train_labels == PNEUMONIA
    mask_other_findings = train_labels == OTHER_FINDING

    # Agora, usamos as máscaras para selecionar as imagens correspondentes
    covid_images = train_images[mask_covid]
    normal_images = train_images[mask_normal]
    pneumonia_images = train_images[mask_pneumonia]
    other_findings_images = train_images[mask_other_findings]

    covid_labels = train_labels[mask_covid]
    normal_labels = train_labels[mask_normal]
    pneumonia_labels = train_labels[mask_pneumonia]
    other_findings_labels = train_labels[mask_other_findings]

    return (covid_images, covid_labels), (normal_images, normal_labels), (pneumonia_images, pneumonia_labels), (other_findings_images, other_findings_labels)


# ================================================

PATH_ROOT = os.getenv('PATH_ROOT')
TEST_STATS_PATH = os.getenv('TEST_STATS_PATH')
TEST_DATASET_PATH = os.getenv('TEST_DATASET_PATH')

# ================================================

test_images = np.load(os.path.join(TEST_DATASET_PATH, 'test_images.npy'))
test_labels = np.load(os.path.join(TEST_DATASET_PATH, 'test_labels.npy'))

(covid_images, covid_labels), (normal_images, normal_labels), (pneumonia_images, pneumonia_labels), (other_findings_images, other_findings_labels) = separete_classes(test_images, test_labels)

pass

model = tf.keras.models.load_model(os.path.join(PATH_ROOT, 'model', 'cnn_model.h5'))

# Testando o modelo
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

print('test_labels.shape: ', test_labels.shape)
print('test_labels.shape: ', predicted_labels.shape)

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
with open(os.path.join(TEST_STATS_PATH, 'stats.json'), 'w') as f:
    json.dump(stats, f)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(TEST_STATS_PATH, 'confusion_matrix.png'))