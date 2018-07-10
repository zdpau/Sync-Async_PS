import numpy as np

train_data = np.load('trainingdata.npz')
images = train_data['images']
labels = train_data['labels']
print(images.shape)
print(labels.shape)
