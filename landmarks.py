from pybug.landmark import LandmarkGroup
from pybug.shape import PointCloud
import numpy as np


def ibug_68_edge(landmark_group):
    group_label = 'ibug_68_edge'

    new_landmarks = PointCloud(landmark_group.lms.points[[33, 36, 45, 8], :])
    new_landmark_group = LandmarkGroup(
        landmark_group._target, group_label, new_landmarks,
        {'all': np.ones(new_landmarks.n_points, dtype=np.bool)})

    new_landmark_group['nose'] = [0]
    new_landmark_group['leye'] = [1]
    new_landmark_group['reye'] = [2]
    new_landmark_group['chin'] = [3]

    return new_landmark_group