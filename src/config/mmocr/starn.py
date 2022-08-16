checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './configs/textrecog/satrn/satrn_academic.pth'
resume_from = None
workflow = [('train', 1)]
label_convertor = dict(
    type='AttnConvertor',
    dict_type='DICT36',
    with_unknown=True,
    lower=True,
    dict_file='/content/drive/MyDrive/AIC/weights/mmocr/vn_dict.txt')
model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=512),
    encoder=dict(
        type='SatrnEncoder',
        n_layers=12,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        n_position=100,
        d_inner=2048,
        dropout=0.1),
    decoder=dict(
        type='NRTRDecoder',
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=2048,
        d_k=64,
        d_v=64),
    loss=dict(type='TFLoss'),
    label_convertor=dict(
        type='AttnConvertor',
        dict_type='DICT36',
        with_unknown=True,
        lower=True,
        dict_file='src/config/mmocr/vn_dict.txt'),
    max_seq_len=25)
dict_file = 'src/config/mmocr/vn_dict.txt'
optimizer = dict(type='Adam', lr=0.0003)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 6
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(
        type='NormalizeOCR',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=100,
                max_width=100,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape'
                ])
        ])
]
dataset_type = 'OCRDataset'
train_prefix = '/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/'
train_ann_file = '/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/train_fold_0'
train = dict(
    type='OCRDataset',
    img_prefix='/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/',
    ann_file='/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/train_fold_0',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
val_prefix = '/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/'
val_ann_file = '/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/val_fold_0'
val = dict(
    type='OCRDataset',
    img_prefix='/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/',
    ann_file='/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/val_fold_0',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train_gen_prefix = 'rec_data_fixbug/'
train_gen_ann_file = 'rec_data_fixbug/rec_gen_train.txt'
train_gen = dict(
    type='OCRDataset',
    img_prefix='rec_data_fixbug/',
    ann_file='rec_data_fixbug/rec_gen_train.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
val_gen_prefix = 'rec_data_fixbug/'
val_gen_ann_file = 'rec_data_fixbug/rec_gen_val.txt'
val_gen = dict(
    type='OCRDataset',
    img_prefix='rec_data_fixbug/',
    ann_file='rec_data_fixbug/rec_gen_val.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
public_prefix = '/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/'
public_ann_file = '/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/rec_public.txt'
public = dict(
    type='OCRDataset',
    img_prefix='/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/',
    ann_file='/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/rec_public.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
private_prefix = '/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/'
private_ann_file = '/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/rec_private.txt'
private = dict(
    type='OCRDataset',
    img_prefix='/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/',
    ann_file=
    '/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/rec_private.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/',
                ann_file=
                '/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/train_fold_0',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='rec_data_fixbug/',
                ann_file='rec_data_fixbug/rec_gen_train.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=100,
                max_width=100,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'text',
                    'valid_ratio', 'resize_shape'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/',
                ann_file=
                '/vinbrain/nthai/AIC2021/mmocr-main/rec_data_fixbug/val_fold_0',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='rec_data_fixbug/',
                ann_file='rec_data_fixbug/rec_gen_val.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=32,
                        min_width=100,
                        max_width=100,
                        keep_aspect_ratio=False,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'resize_shape'
                        ])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='../det_result/exp',
                ann_file=
                '../det_result/exp/test_list.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            # dict(
            #     type='OCRDataset',
            #     img_prefix=
            #     '/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/',
            #     ann_file=
            #     '/vinbrain/nthai/AIC2021/mmocr-main/rec_TestA_new/rec_private.txt',
            #     loader=dict(
            #         type='HardDiskLoader',
            #         repeat=1,
            #         parser=dict(
            #             type='LineStrParser',
            #             keys=['filename', 'text'],
            #             keys_idx=[0, 1],
            #             separator=' ')),
            #     pipeline=None,
            #     test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=32,
                        min_width=100,
                        max_width=100,
                        keep_aspect_ratio=False,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'resize_shape'
                        ])
                ])
        ]))
evaluation = dict(interval=1, metric='acc')
work_dir = '5fold/fold0'
gpu_ids = range(0)
