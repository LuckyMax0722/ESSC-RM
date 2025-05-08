semantic_kitti_class_frequencies = [
        5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
        8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
        4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
        1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
    ]

# 20 classes with unlabeled
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign',
]
num_class = len(class_names)

"""Model params."""
dim = 32

text_encoder_dim = {
    'CLIP' : 512,
    'JinaCLIP' : 512,
    'LongCLIP' : 768,
    'JinaCLIP_1024' : 1024,
}

model_text = dict(
    num_heads=8,
)

"""Training params."""
learning_rate=5e-5
training_steps=25000

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.01
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    frequency=1
)
