import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from skimage.transform import rescale
from constants import DATA_DIR, SCALE_PARAMETER, IMAGE_LIMIT, N_COMPONENTS, SVD_SOLVER, WHITEN
from eigenface import Eigenface



eigenface = Eigenface()
face_matrix = eigenface.read_data(DATA_DIR, SCALE_PARAMETER, IMAGE_LIMIT)
face_matrix, mean_face = eigenface.normalize(face_matrix)
covariance_matrix = eigenface.get_covariance_matrix(face_matrix)
# pca = eigenface.get_pca(covariance_matrix, N_COMPONENTS, SVD_SOLVER, WHITEN)
svd = eigenface.get_svd(covariance_matrix, N_COMPONENTS)

eigenface.recognize_face("yalefaces/subject01.happy", mean_face, SCALE_PARAMETER, svd)


"""
Classify an image to one of the eigenfaces.
"""
#
# def classify(self, path_to_img):
#     img = cv2.imread(path_to_img, 0)  # read as a grayscale image
#     img_col = np.array(img, dtype='float64').flatten()  # flatten the image
#     img_col -= self.mean_img_col  # subract the mean column
#     img_col = np.reshape(img_col, (self.mn, 1))  # from row vector to col vector
#
#     S = self.evectors.transpose() * img_col  # projecting the normalized probe onto the
#     # Eigenspace, to find out the weights
#
#     diff = self.W - S  # finding the min ||W_j - S||
#     norms = np.linalg.norm(diff, axis=0)
#
#     closest_face_id = np.argmin(norms)  # the id [0..240) of the minerror face to the sample
#     return (closest_face_id / self.train_faces_count) + 1