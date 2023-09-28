from transforms import MergeLandmarks, FlattenLandmarks, Randomize, Padding
from sign_language_tools.pose.transform import (
    HorizontalFlip,
    GaussianNoise,
    RandomRotation2D,
    RandomTranslation,
    Compose,
)


def get_skeleton_transforms(seq_size):
    """
    Return a simple transform for skeletons and hands landmarks without data augmentation
    """

    merge = MergeLandmarks(
        {"pose": [0, 23], "left_hand": [None, None], "right_hand": [None, None]}
    )
    padding = Padding(seq_size)
    flatten = FlattenLandmarks()

    return Compose([merge, padding, flatten])


def get_data_augmentation_transforms(seq_size):
    """
    Return a simple transform for skeletons and hands landmarks with data augmentation
    """

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

    transforms.append(Padding(seq_size))
    transforms.append(FlattenLandmarks())

    return Compose(transforms)
