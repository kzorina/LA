import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import rescale
from sklearn.decomposition import PCA, TruncatedSVD


class Eigenface:
    @staticmethod
    def get_mean(face_matrix):
        return face_matrix.mean(axis=0)

    @staticmethod
    def normalize(face_matrix, mean_face):
        for column in face_matrix:
            column = column.astype('float64')
            column -= mean_face

    @staticmethod
    def read_data(file_path, scale_parameter):
        face_matrix = []

        for file in os.listdir(file_path):
            image_path = os.path.join(file_path, file)
            image = np.asarray(Image.open(image_path))
            image_vector = rescale(image, scale_parameter, mode='reflect').flatten()
            face_matrix.append(image_vector)

        face_matrix = np.array(face_matrix)
        return face_matrix

    @staticmethod
    def get_covariance_matrix(face_matrix):
        return np.dot(face_matrix.T, face_matrix)

    @staticmethod
    def get_pca(covariance_matrix, n_components, svd_solver, whiten):
        t0 = time()
        pca = PCA(n_components=n_components,
                  svd_solver=svd_solver,
                  whiten=whiten).fit(covariance_matrix)
        print("Sklearn PCA done in {} seconds".format(time() - t0))
        return pca

    @staticmethod
    def get_svd(covariance_matrix, n_components):
        t0 = time()
        svd = TruncatedSVD(n_components=n_components,
                           random_state=1).fit(covariance_matrix)
        print("Found {} first elements using scikit SVD in {} seconds".format(n_components,
                                                                              time() - t0))
        return svd

    @staticmethod
    def decompose_face(image_path, mean_face, scale_parameter, svd):
        image = np.asarray(Image.open(image_path)).astype('float64')
        image_vector = rescale(image, scale_parameter, mode='reflect').flatten()
        image_vector = image_vector - mean_face  # normalize
        image_weights = []
        for eigenvector in svd.components_:
            coefficient = np.dot(eigenvector.transpose(), image_vector)
            image_weights.append(coefficient)
        return image_weights

    def train_model(self, train_dir, mean_face, scale_parameter, svd):
        decomposed_images = {}
        for file in os.listdir(train_dir):
            image_path = os.path.join(train_dir, file)
            image_weights = self.decompose_face(image_path, mean_face, scale_parameter, svd)
            decomposed_images[image_path] = image_weights
        return decomposed_images

    def recognise_image(self, known_images, image_path, mean_face, scale_parameter, svd):
        image_weights = self.decompose_face(image_path, mean_face, scale_parameter, svd)

        distances = {}
        for filepath, weights in known_images.items():
            distance = 0
            for index, weight in enumerate(weights):
                distance += np.square(image_weights[index] - weight) / svd.singular_values_[
                    index]
            distances[filepath] = distance

        min_distance = min(distances.values())
        for filepath, distance in distances.items():
            if np.isclose(distance, min_distance):
                image_match = np.asarray(Image.open(filepath))
                image_input = np.asarray(Image.open(image_path))
                shape = image_match.shape

                plt.subplot(1, 2, 1)
                plt.imshow(image_input.reshape(shape[0], shape[1]), cmap=plt.cm.gray)
                plt.title("Input")

                plt.subplot(1, 2, 2)
                plt.imshow(image_match.reshape(shape[0], shape[1]), cmap=plt.cm.gray)
                plt.title("Match\ndistance: {}".format(distance))

                plt.show()
                break

    def recognize_test_images(self, test_dir, mean_face, scale_parameter, svd, known_images):
        for file in os.listdir(test_dir):
            image_path = os.path.join(test_dir, file)
            self.recognise_image(known_images, image_path, mean_face, scale_parameter, svd)
