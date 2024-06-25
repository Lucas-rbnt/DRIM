from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    SpatialPadd,
    CopyItemsd,
    OneOf,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    ToTensord,
    NormalizeIntensity,
    SpatialPad,
)


def get_tumor_transforms(roi_size):
    return Compose(
        [
            NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),
            SpatialPadd(keys=["image"], spatial_size=roi_size),
            CopyItemsd(
                keys=["image"], times=1, names=["image_2"], allow_missing_keys=False
            ),
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=8,
                        spatial_size=10,
                        dropout_holes=True,
                        max_spatial_size=18,
                    ),
                    RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=6,
                        spatial_size=18,
                        dropout_holes=False,
                        max_spatial_size=24,
                    ),
                ]
            ),
            RandCoarseShuffled(
                keys=["image"], prob=0.8, holes=4, spatial_size=4, max_spatial_size=4
            ),
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=8,
                        spatial_size=10,
                        dropout_holes=True,
                        max_spatial_size=18,
                    ),
                    RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=6,
                        spatial_size=18,
                        dropout_holes=False,
                        max_spatial_size=24,
                    ),
                ]
            ),
            RandCoarseShuffled(
                keys=["image_2"], prob=0.8, holes=10, spatial_size=8, max_spatial_size=8
            ),
            ToTensord(keys=["image", "image_2"]),
        ]
    )


tumor_transfo = Compose(
    [
        NormalizeIntensity(nonzero=True, channel_wise=True),
        SpatialPad(spatial_size=(64, 64, 64)),
    ]
)
