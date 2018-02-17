import os

from constants import *
from eigenface import Eigenface

eigenface = Eigenface()
face_matrix = eigenface.read_data(DATA_DIR, SCALE_PARAMETER)
face_matrix, mean_face = eigenface.normalize(face_matrix)
covariance_matrix = eigenface.get_covariance_matrix(face_matrix)
svd = eigenface.get_svd(covariance_matrix, N_COMPONENTS)

decomposed_images = eigenface.train_model(TRAIN_DIR, mean_face, SCALE_PARAMETER, svd, STANDARD_SIZE)

eigenface.recognize_test_images(decomposed_images, mean_face, SCALE_PARAMETER, svd, TEST_DIR, STANDARD_SIZE)
