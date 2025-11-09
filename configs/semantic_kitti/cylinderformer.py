_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="GlobalSemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Tester
test = dict(type="SemSegGlobalTester", verbose=True)

# model settings
model = dict(
    type="DefaultSegmentorGlobal",
    num_classes=19,
    backbone_out_channels=64, # 64,
    backbone=dict(
        type="cylinderformer",
        in_channels=4,
        order=["z", "z-q90", "z-q180",
               "hilbert", "hilbert-q90", "hilbert-q270",
               "z-trans", "z-trans-q90", "z-trans-q270",
               "hilbert-trans", "hilbert-trans-q90"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             weight=[3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704, 10.1922, 1.6155, 4.2187,
                     1.9385, 5.5455, 2.0198, 2.6261, 1.3212, 5.1102, 2.5492, 5.8585, 7.3929],
             loss_weight=1.0,
             ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "SemanticKITTIGlobalFilterDataset"
dataset_val_type = "SemanticKITTIGlobalFilterTestDataset"
data_root = "/dataset/sequences/"
ignore_index = -1
names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

data = dict(
    num_classes=19,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="PolarGlobalSample",
                hash_type="fnv",
                mode="train",
            ),
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="SphereGlobalCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereGlobalCrop", point_max=120000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("grid_all", "segment_all", "com_mask", "mask_quadrant"),
                offset_keys_dict=dict(offset="coord_all"),
                feat_keys=("coord_all", "strength_all"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="PolarGlobalSample",
                hash_type="fnv",
                mode="train",
            ),
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("grid_all", "segment_all", "com_mask", "mask_quadrant"),
                offset_keys_dict=dict(offset="coord_all"),
                feat_keys=("coord_all", "strength_all"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_val_type,
        split="val",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="PolarGlobalSample",
                hash_type="fnv",
                mode="test",
            ),
            crop=None,
            post_transform=[
                # dict(
                #     type="PointClip",
                #     point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                # ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("grid_all", "com_mask", "mask_quadrant", "index"),
                    offset_keys_dict=dict(offset="coord_all"),
                    feat_keys=("coord_all", "strength_all"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
