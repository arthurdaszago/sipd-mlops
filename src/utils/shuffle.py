import numpy as np

def shuffle_in_order(images, labels):
    print('len(images): ', images.shape)
    print('len(labels): ', labels.shape)

    assert len(images) == len(labels)  # ensure arrays are of the same length
    p = np.random.permutation(len(images))
    return images[p], labels[p]