from constants import *
from eigenface import Eigenface

eigenface = Eigenface()
face_matrix = eigenface.read_data(DATA_DIR, SCALE_PARAMETER)
mean_face = eigenface.get_mean(face_matrix)
eigenface.normalize(face_matrix, mean_face)
covariance_matrix = eigenface.get_covariance_matrix(face_matrix)
svd = eigenface.get_svd(covariance_matrix, N_COMPONENTS)
svd.components_ = svd.components_[N_OF_EVCS_TO_DROP:]

decomposed_images = eigenface.train_model(TRAIN_DIR, mean_face, SCALE_PARAMETER, svd, STANDARD_SIZE)
eigenface.recognize_test_images(TEST_DIR, mean_face, SCALE_PARAMETER, svd, decomposed_images, STANDARD_SIZE)

