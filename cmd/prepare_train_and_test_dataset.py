import numpy as np
import tensorflow as tf
import os

PATH_ROOT = None
if os.getenv('PATH_ROOT') is not None:
    PATH_ROOT = os.getenv('PATH_ROOT')
else:
    PATH_ROOT = '/home/arthur/Documents/ifc/tc/code/sipd-mlops'

# Carregar o dataset CIFAR-10
(train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()

# Definindo os índices das classes no CIFAR-10
AVIÃO, AUTOMOBILE, PÁSSARO, GATO, VEADO, CACHORRO, SAPO, CAVALO, NAVIO, CAMINHÃO = range(10)

# Classes de interesse
individual_classes = [AVIÃO, CAMINHÃO, NAVIO]
other_classes = [PÁSSARO, GATO, VEADO, CACHORRO, CAVALO]
excluded_class = [AUTOMOBILE]

# Preparando as imagens e rótulos
mask_individual = np.isin(train_labels, individual_classes)
mask_others = np.isin(train_labels, other_classes)

images_individual = train_images[mask_individual.squeeze()]
labels_individual = np.array([individual_classes.index(label[0]) for label in train_labels[mask_individual.squeeze()]]).reshape(-1, 1)  # Convertendo rótulos para o novo sistema

images_others = train_images[mask_others.squeeze()]
labels_others = np.full(shape=images_others.shape[0], fill_value=3).reshape(-1, 1)  # Label 3 para "outros"

# Combinando as imagens e rótulos
train_images_final = np.vstack([images_individual, images_others])
train_labels_final = np.vstack([labels_individual, labels_others])

# Máscara para a classe AUTOMOBILE
mask_automobile = (train_labels == AUTOMOBILE)

images_automobile = train_images[mask_automobile.squeeze()]
labels_automobile = train_labels[mask_automobile.squeeze()]

# print(images_automobile.shape)
# print(labels_automobile.shape)

# Embaralhando e separando 10% para validação
p = np.random.permutation(train_images_final.shape[0])
train_images_final = train_images_final[p]
train_labels_final = train_labels_final[p]

test_images = train_images_final[:4000]
test_labels = train_labels_final[:4000]

train_images = train_images_final[4000:]
train_labels = train_labels_final[4000:]

# Verificar o número de amostras por classe
num_samples_per_class = []

for class_index in range(10):
    num_samples = np.sum(train_labels == class_index)
    num_samples_per_class.append(num_samples)
    class_name = [
        'AVIÃO', 'AUTOMOBILE', 'PÁSSARO', 'GATO', 'VEADO',
        'CACHORRO', 'SAPO', 'CAVALO', 'NAVIO', 'CAMINHÃO'
    ][class_index] 
    print(f"Number of samples for {class_name}: {num_samples}")

# Salvando os datasets
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_automobile_images.npy'), images_automobile)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_automobile_labels.npy'), labels_automobile)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_images.npy'), train_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_labels.npy'), train_labels)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_images.npy'), test_images)
np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_labels.npy'), test_labels)

print("Datasets de treino e validação preparados.")

# ========================================
# ========================================
# ========================================
# ========================================

# import numpy as np
# import tensorflow as tf
# import os

# PATH_ROOT = None
# if os.getenv('PATH_ROOT') is not None:
#     PATH_ROOT = os.getenv('PATH_ROOT')
# else:
#     PATH_ROOT = '/home/arthur/Documents/ifc/tc/code/sipd-mlops'

# # Carregar o dataset CIFAR-10
# (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()

# # Definindo os índices das classes no CIFAR-10
# AVIÃO, AUTOMOBILE, PÁSSARO, GATO, VEADO, CACHORRO, SAPO, CAVALO, NAVIO, CAMINHÃO = range(10)

# # Classes de interesse
# individual_classes = [AVIÃO, CAMINHÃO, NAVIO]
# other_classes = [PÁSSARO, GATO, VEADO, CACHORRO, CAVALO]
# excluded_class = [AUTOMOBILE]

# # Determinando o número mínimo de amostras
# min_samples = min([np.sum(train_labels == cls) for cls in individual_classes])
# min_samples_other = min_samples // len(other_classes)

# images_list = []
# labels_list = []

# # Coletando amostras das classes individuais
# for label in individual_classes:
#     mask = (train_labels == label).squeeze()
#     samples = train_images[mask][:min_samples]
#     images_list.append(samples)
#     labels_list.append(np.full((min_samples, 1), individual_classes.index(label)))

# # Coletando amostras da classe "outros"
# for label in other_classes:
#     mask = (train_labels == label).squeeze()
#     samples = train_images[mask][:min_samples_other]
#     images_list.append(samples)
#     labels_list.append(np.full((min_samples_other, 1), 3))  # Label 3 para "outros"

# # Combinando as imagens e rótulos
# train_images_final = np.vstack(images_list)
# train_labels_final = np.vstack(labels_list)

# # Máscara para a classe AUTOMOBILE
# mask_automobile = (train_labels == AUTOMOBILE).squeeze()

# images_automobile = train_images[mask_automobile][:min_samples]
# labels_automobile = train_labels[mask_automobile][:min_samples]

# # Embaralhando e separando 10% para validação
# p = np.random.permutation(train_images_final.shape[0])
# train_images_final = train_images_final[p]
# train_labels_final = train_labels_final[p]

# test_images = train_images_final[:4000]
# test_labels = train_labels_final[:4000]

# train_images = train_images_final[4000:]
# train_labels = train_labels_final[4000:]

# # # Verificar o número de amostras por classe
# num_samples_per_class = []

# for class_index in range(10):
#     num_samples = np.sum(train_labels == class_index)
#     num_samples_per_class.append(num_samples)
#     class_name = [
#         'AVIÃO', 'AUTOMOBILE', 'PÁSSARO', 'GATO', 'VEADO',
#         'CACHORRO', 'SAPO', 'CAVALO', 'NAVIO', 'CAMINHÃO'
#     ][class_index] 
#     print(f"Number of samples for {class_name}: {num_samples}")

# # Salvando os datasets
# np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_automobile_images.npy'), images_automobile)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_automobile_labels.npy'), labels_automobile)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_images.npy'), train_images)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_labels.npy'), train_labels)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_images.npy'), test_images)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_labels.npy'), test_labels)

# print("Datasets de treino e validação balanceados preparados.")

# =========================
# =========================
# =========================
# =========================

# import numpy as np
# import tensorflow as tf
# import os

# PATH_ROOT = None
# if os.getenv('PATH_ROOT') is not None:
#     PATH_ROOT = os.getenv('PATH_ROOT')
# else:
#     PATH_ROOT = '/home/arthur/Documents/ifc/tc/code/sipd-mlops'

# # Carregar o dataset CIFAR-10
# (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()

# # Definindo os índices das classes no CIFAR-10
# AVIÃO, AUTOMOBILE, PÁSSARO, GATO, VEADO, CACHORRO, SAPO, CAVALO, NAVIO, CAMINHÃO = range(10)

# # Classes de interesse
# individual_classes = [AVIÃO, CAMINHÃO, NAVIO]
# other_classes = [PÁSSARO, GATO, VEADO, CACHORRO, CAVALO]
# excluded_class = [AUTOMOBILE]

# # Função para equilibrar amostras
# def balance_samples(images, labels, num_samples):
#     indices = np.random.choice(images.shape[0], num_samples, replace=False)
#     return images[indices], labels[indices]

# # Preparando as imagens e rótulos
# mask_individual = np.isin(train_labels, individual_classes)
# images_individual = train_images[mask_individual.squeeze()]
# labels_individual = np.array([individual_classes.index(label[0]) for label in train_labels[mask_individual.squeeze()]]).reshape(-1, 1)

# mask_others = np.isin(train_labels, other_classes)
# images_others = train_images[mask_others.squeeze()]
# labels_others = np.full(shape=images_others.shape[0], fill_value=3).reshape(-1, 1)  # Label 3 para "outros"

# # Balanceando as amostras
# images_individual, labels_individual = balance_samples(images_individual, labels_individual, 5000)
# images_others, labels_others = balance_samples(images_others, labels_others, 5000)

# # Combinando as imagens e rótulos
# train_images_final = np.vstack([images_individual, images_others])
# train_labels_final = np.vstack([labels_individual, labels_others])

# # Máscara para a classe AUTOMOBILE
# mask_automobile = (train_labels == AUTOMOBILE)

# images_automobile = train_images[mask_automobile][:min_samples]
# labels_automobile = train_labels[mask_automobile][:min_samples]

# images_automobile = train_images[mask_automobile.squeeze()]
# labels_automobile, _ = balance_samples(labels_automobile, labels_automobile, 5000)

# # Embaralhando e separando 10% para validação
# p = np.random.permutation(train_images_final.shape[0])
# train_images_final = train_images_final[p]
# train_labels_final = train_labels_final[p]

# test_images = train_images_final[:4000]
# test_labels = train_labels_final[:4000]

# train_images = train_images_final[4000:]
# train_labels = train_labels_final[4000:]

# # Salvando os datasets
# np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_automobile_images.npy'), images_automobile)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_automobile_labels.npy'), labels_automobile)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_images.npy'), train_images)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'train', 'train_labels.npy'), train_labels)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_images.npy'), test_images)
# np.save(os.path.join(PATH_ROOT, 'datasets', 'test', 'test_labels.npy'), test_labels)

# print("Datasets de treino e validação preparados.")

