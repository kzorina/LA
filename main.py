import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from skimage.transform import rescale

# %matplotlib inline

#DEFAULT_IMAGE_SIZE = 77760
#DEFAULT_IMAGE_SIZE = 9520
DATA_DIR = 'yalefaces'
SCALE_PARAMETER = 0.25
IMAGE_LIMIT = 20

face_matrix = []
for index, file in enumerate(os.listdir(DATA_DIR)):
    image_path = os.path.join(DATA_DIR, file)
    image = np.asarray(Image.open(image_path))
    image = rescale(image, SCALE_PARAMETER, mode='reflect')
    image_vector = image.flatten()
    face_matrix.append(image_vector)
    if index > IMAGE_LIMIT:
        break

image_shape = image.shape
face_matrix = np.array(face_matrix)

mean_face = face_matrix.mean(axis=0)
# plt.imshow(mean_face.reshape(image_shape[0], image_shape[1]), cmap=plt.cm.gray)
for column in face_matrix:
    column = column.astype('float64')
    column -= mean_face

cov_face_matrix = np.dot(face_matrix.T,face_matrix)

t0 = time()
u, s, v = np.linalg.svd(cov_face_matrix)
print("Numpy SVD done in {} seconds".format(time() - t0))

t0 = time()
pca = PCA(svd_solver="auto", whiten=True).fit(cov_face_matrix)
print("Sklearn PCA done in {} seconds".format(time() - t0))
print(pca.explained_variance_ratio_)



