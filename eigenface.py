import os
from time import time

import numpy as np
from PIL import Image
from skimage.transform import rescale
from sklearn.decomposition import PCA, TruncatedSVD

class Eigenface:
    @staticmethod
    def normalize(face_matrix):
        mean_face = face_matrix.mean(axis=0)
        for column in face_matrix:
            column = column.astype('float64')
            column -= mean_face
        return face_matrix, mean_face

    @staticmethod
    def read_data(file_path, scale_parameter, image_limit):
        face_matrix = []

        for index, file in enumerate(os.listdir(file_path)):
            image_path = os.path.join(file_path, file)
            image = np.asarray(Image.open(image_path))
            image_vector = rescale(image, scale_parameter, mode='reflect').flatten()
            face_matrix.append(image_vector)
            if index > image_limit:
                break

        face_matrix = np.array(face_matrix)
        return face_matrix

    @staticmethod
    def get_covariance_matrix(face_matrix):
        return np.dot(face_matrix.T, face_matrix)

    @staticmethod
    def get_pca(covariance_matrix, n_components, svd_solver, whiten):
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten).fit(covariance_matrix)
        print("Sklearn PCA done in {} seconds".format(time() - t0))
        return pca


    @staticmethod
    def get_svd(covariance_matrix, n_components):
        t0 = time()
        svd = TruncatedSVD(n_components=n_components, random_state=1).fit(covariance_matrix)
        print("Scikit SVD done in {} seconds".format(time() - t0))
        return svd

    @staticmethod
    def decompose_face(image_path, mean_face, scale_parameter, svd):
        image = np.asarray(Image.open(image_path))
        image_vector = rescale(image, scale_parameter, mode='reflect').flatten()
        image_vector = image_vector - mean_face  # normalize
        image_vector_hat = []
        for eigenvector in svd.components_:
            coefficient = np.dot(eigenvector.transpose(), image_vector)
            image_vector_hat.append(coefficient)
        print(image_vector_hat)
        return image_path, image_vector_hat



