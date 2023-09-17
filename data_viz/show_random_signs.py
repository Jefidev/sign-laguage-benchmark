from sign_language_tools.visualization import VideoPlayer
from sign_language_tools.pose.mediapipe.edges import (
    UPPER_POSE_EDGES,
    HAND_EDGES,
    LIPS_EDGES,
    FACE_EDGES,
)
import numpy as np

from random import sample


def show_random_signs(data, num_signs=10, video=True):
    instances = sample(data.instances, num_signs)

    for instance in instances:
        id_gloss = data.targets[instance]
        print(data.index_to_label[id_gloss] + "\n")

        player = VideoPlayer(
            root=data.config.root, screenshot_dir="/home/jeromefink/Images", fps=50
        )

        landmarks = load_landmarks(data, instance)

        player.attach_pose("Pose", landmarks["pose"], connections=UPPER_POSE_EDGES)
        player.attach_pose("Left hand", landmarks["left_hand"], connections=HAND_EDGES)
        player.attach_pose(
            "Right hand", landmarks["right_hand"], connections=HAND_EDGES
        )

        if video:
            player.attach_video(f"videos/{instance}.mp4")

        player.set_speed(0.3)
        player.play()


def load_landmarks(data, instance_id):
    pose_folder = "poses_raw" if data.config.use_raw else "poses"

    coordinate_indices = [0, 1, 2] if data.config.use_3d else [0, 1]
    max_len = data.config.sequence_max_length

    instance_features = {}

    for landmark_set in data.config.landmarks:
        filepath = f"{data.config.root}/{pose_folder}/{landmark_set}/{instance_id}.npy"
        lm_set_features = np.load(filepath)[:, :, coordinate_indices]
        instance_features[landmark_set] = lm_set_features

    if data.config.transform is not None:
        instance_features = data.config.transform(instance_features)

    return instance_features
