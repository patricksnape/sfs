from pybug.shape import PointCloud
from pybug.transform import SimilarityTransform
import numpy as np


def build_similarity_transform(shape):
    L_E_X = shape.points[1, 1]
    L_E_Y = shape.points[1, 0]

    R_E_X = shape.points[2, 1]
    R_E_Y = shape.points[2, 0]

    Nose_X = shape.points[0, 1]
    Nose_Y = shape.points[0, 0]

    Low_X = shape.points[3, 1]
    Low_Y= shape.points[3, 0]

    original_landmarks = np.array([[L_E_Y, L_E_X],
                                   [Nose_Y, Nose_X],
                                   [Low_Y, Low_X],
                                   [R_E_Y, R_E_X]])

    # Assumes a template size of 480x360
    # Build the template face (xs)
    L_E_X = 24
    R_E_X = 336
    Nose_X = 180
    Low_X = 180

    # Build the template face (ys)
    L_E_Y = 112
    R_E_Y = 112
    Nose_Y = 282
    Low_Y = 452

    face_template = np.array([[L_E_Y, L_E_X],
                              [Nose_Y, Nose_X],
                              [Low_Y, Low_X],
                              [R_E_Y, R_E_X]])
    return SimilarityTransform.align(PointCloud(face_template),
                                     PointCloud(original_landmarks))