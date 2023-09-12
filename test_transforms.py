from lsfb_dataset.datasets.lsfb_isol import LSFBIsolConfig, LSFBIsolLandmarks
from data_viz import show_random_signs
from sign_language_tools.pose.transform import (
    HorizontalFlip,
    GaussianNoise,
    RandomRotation2D,
    RandomTranslation,
)

from sign_language_tools.common.transforms import Compose

from transforms import MergeLandmarks, SplitLandmarks, Randomize

DS_ROOT = "/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/lsfb_isol"

# Composition of transforms

transforms = []
transforms.append(
    MergeLandmarks(
        {"pose": [0, 23], "left_hand": [None, None], "right_hand": [None, None]}
    )
)

transforms.append(Randomize(GaussianNoise(0.002)))
transforms.append(Randomize(HorizontalFlip(), 0.2))
transforms.append(Randomize(RandomRotation2D(angle_range=(-0.3, 0.3))))
transforms.append(Randomize(RandomTranslation()))

transforms.append(
    SplitLandmarks({"pose": [0, 23], "left_hand": [23, 44], "right_hand": [44, 65]})
)

composed_transforms = Compose(transforms)


# Dataset configuration

config = LSFBIsolConfig(
    root="/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/lsfb_isol",
    split="mini_sample",
    n_labels=2000,
    landmarks=["pose", "left_hand", "right_hand"],
    transform=composed_transforms,
    sequence_max_length=50,
)

test_dataset = LSFBIsolLandmarks(config)

show_random_signs(test_dataset, num_signs=20)
