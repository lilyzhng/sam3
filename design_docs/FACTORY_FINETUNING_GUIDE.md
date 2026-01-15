# SAM3 Factory Fine-Tuning Guide

A comprehensive guide for fine-tuning SAM3 (Segment Anything 3) for factory and industrial applications. This guide covers a two-stage training approach: **spatial adaptation** (images) followed by **temporal adaptation** (video tracking).

---

## 🚀 Execution Order (Start Here!)

This guide is organized into **phases**. Complete each phase before moving to the next.

### Phase 0: Pipeline Validation (Do This First!)

**Goal:** Verify the training pipeline works before using your own data.

| Step | Action | Dataset | Config |
|------|--------|---------|--------|
| 0.1 | Download Roboflow-100 | [Roboflow-100](https://universe.roboflow.com/roboflow-100) | Existing SAM3 config |
| 0.2 | Run Stage 1 training | Roboflow-100 (images) | `roboflow_v100_full_ft_100_images.yaml` |
| 0.3 | Verify training completes | Check loss curves, checkpoints | — |

**Why Roboflow-100?** SAM3 already has working configs for this dataset. No data conversion needed.

```bash
# Example command for Phase 0
python sam3/train/train.py -c sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 0 --num-gpus 1
```

**✅ Success criteria:** Training runs without errors, loss decreases, checkpoint is saved.

---

### Phase 1: Two-Stage Training with YouTube-VOS

**Goal:** Validate the full two-stage (spatial → temporal) pipeline.

**⚠️ Important:** Both stages must use the **same dataset domain**. Otherwise the temporal stage can't track objects the spatial stage didn't learn.

| Step | Action | Dataset | Format |
|------|--------|---------|--------|
| 1.1 | Download YouTube-VOS | [YouTube-VOS](https://youtube-vos.org/dataset/vos/) | YTVIS-style |
| 1.2 | Extract frames for Stage 1 | YouTube-VOS frames | Convert to COCO format |
| 1.3 | Run Stage 1 (spatial) | YouTube-VOS frames as images | Custom config |
| 1.4 | Run Stage 2 (temporal) | YouTube-VOS full videos | Custom config (loads Stage 1 checkpoint) |

**Why YouTube-VOS?** 
- SAM3 was NOT trained on it (unlike SA-V)
- Has both frames (for Stage 1) AND video sequences (for Stage 2)
- Same objects appear in both stages → proper transfer learning

---

### Phase 2: Factory Domain Adaptation

**Goal:** Fine-tune for your specific factory environment.

| Step | Action | Dataset |
|------|--------|---------|
| 2.1 | Collect factory images | 300-500 images with masks |
| 2.2 | Collect factory videos | 50-100 video clips with tracking annotations |
| 2.3 | Run Stage 1 (spatial) | Factory images |
| 2.4 | Run Stage 2 (temporal) | Factory videos |

**⚠️ Key insight:** For real factory deployment, you need factory-domain data for BOTH stages.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 0: Pipeline Validation with Roboflow-100](#phase-0-pipeline-validation-with-roboflow-100)
4. [Phase 1: Two-Stage Training with YouTube-VOS](#phase-1-two-stage-training-with-youtube-vos)
5. [Phase 2: Factory Domain Adaptation](#phase-2-factory-domain-adaptation)
6. [Data Preparation Scripts](#data-preparation-scripts)
7. [Training Commands](#training-commands)
8. [Monitoring & Evaluation](#monitoring--evaluation)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Overview

### Why Two Stages?

SAM3 has two main capabilities:
1. **Spatial understanding** - Detecting and segmenting objects in images
2. **Temporal understanding** - Tracking objects across video frames

Training these separately ensures:
- Stage 1 teaches the model **what** to segment in your target domain
- Stage 2 teaches the model **how to track** those objects over time

**⚠️ Critical:** Both stages must use data from the **same domain**. Using different datasets (e.g., Roboflow for Stage 1, YouTube-VOS for Stage 2) will NOT work because the model learns different objects in each stage.

### Training Strategy Summary

| Stage | Goal | Data | Epochs | LR Scale |
|-------|------|------|--------|----------|
| **Stage 1** | Spatial adaptation | 300-500 images with masks | 40 | 0.1 |
| **Stage 2** | Temporal adaptation | 50-100 video clips | 20 | 0.05 |

### Key Techniques Used

- **Full fine-tuning** (not LoRA/PEFT) - SAM3's approach
- **Differential learning rates** - Higher LR for decoder, lower for pretrained backbones
- **Layer-wise LR decay** - Earlier layers learn slower
- **Mixed precision (BF16)** - Faster training, lower memory
- **Activation checkpointing** - Reduces GPU memory usage

### Dataset Selection Guide

| Dataset | SAM3 trained on it? | Use for fine-tuning? | Notes |
|---------|---------------------|----------------------|-------|
| SA-V | ✅ Yes | ❌ No | Already seen during SAM3 training |
| SA-Co/VEval | ✅ Yes | ❌ No | Use for evaluation only |
| Roboflow-100 | ❌ No | ✅ Yes (Stage 1 only) | Good for pipeline validation |
| YouTube-VOS | ❌ No | ✅ Yes (both stages) | Best for two-stage validation |
| Your factory data | ❌ No | ✅ Yes (both stages) | Required for production |

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | 1x 24GB (A10/3090) | 1x 40GB+ (A100) |
| RAM | 32GB | 64GB |
| Storage | 50GB free | 100GB+ free |
| Python | 3.12+ | 3.12 |
| PyTorch | 2.7+ | 2.7 |
| CUDA | 12.6+ | 12.6 |

### Installation

```bash
# Clone and enter the repository
cd /path/to/sam3_repo

# Create conda environment
conda create -n sam3 python=3.12
conda deactivate && conda activate sam3

# Install PyTorch with CUDA
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install SAM3 with training dependencies
pip install -e ".[train]"
```

### Authenticate with Hugging Face

SAM3 checkpoints require access approval:

```bash
# Generate token at https://huggingface.co/settings/tokens
huggingface-cli login
```

---

## Phase 0: Pipeline Validation with Roboflow-100

**⏱️ Time required:** 2-4 hours (including download and training)

### Goal

Validate that the SAM3 training pipeline works on your system before investing time in data preparation.

### Step 0.1: Download Roboflow-100

Roboflow-100 is a collection of 100 diverse datasets. For validation, we'll use a single subset.

```bash
# Create data directory
mkdir -p data/roboflow100

# Option A: Download via Roboflow CLI (recommended)
pip install roboflow
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')  # Get from https://roboflow.com/settings/api
project = rf.workspace('roboflow-100').project('cells-uyemf')  # Example: cells dataset
dataset = project.version(2).download('coco', location='data/roboflow100/cells')
"

# Option B: Download manually from https://universe.roboflow.com/roboflow-100
# Look for datasets with 'coco' export format
```

### Step 0.2: Update Config Paths

Edit `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`:

```yaml
paths:
  roboflow_base_path: /absolute/path/to/data/roboflow100
  experiment_log_dir: /absolute/path/to/experiments/roboflow_validation
```

### Step 0.3: Run Training

```bash
# Single GPU training
python sam3/train/train.py \
    -c sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 0 \
    --num-gpus 1

# Monitor training (in another terminal)
tail -f /path/to/experiments/roboflow_validation/logs/*.log
```

### Step 0.4: Verify Success

Check for:
- [ ] Training starts without errors
- [ ] Loss decreases over epochs
- [ ] Checkpoint files are created in `checkpoints/` directory
- [ ] Validation metrics are logged

**✅ If all checks pass:** Proceed to Phase 1 or Phase 2.

**❌ If training fails:** See [Troubleshooting](#troubleshooting) section.

---

## Phase 1: Two-Stage Training with YouTube-VOS

**⏱️ Time required:** 1-2 days (including download, conversion, and training)

### Goal

Validate the full two-stage pipeline using a dataset SAM3 hasn't seen.

### Step 1.1: Download YouTube-VOS

```bash
# Create data directory
mkdir -p data/youtube_vos

# Download from https://youtube-vos.org/dataset/vos/
# You'll need to register and accept the license
# Download: train.zip, valid.zip

# Extract
unzip train.zip -d data/youtube_vos/
unzip valid.zip -d data/youtube_vos/
```

Expected structure:
```
data/youtube_vos/
├── train/
│   ├── JPEGImages/
│   │   ├── video_001/
│   │   │   ├── 00000.jpg
│   │   │   ├── 00005.jpg
│   │   │   └── ...
│   │   └── video_002/
│   └── Annotations/
│       ├── video_001/
│       │   ├── 00000.png
│       │   └── ...
│       └── video_002/
└── valid/
    └── ...
```

### Step 1.2: Convert Frames to COCO Format (Stage 1)

```bash
# Run conversion script (see Data Preparation Scripts section)
python scripts/convert_ytvos_to_coco.py \
    --input-dir data/youtube_vos/train \
    --output-dir data/youtube_vos_images \
    --sample-rate 5  # Take every 5th frame
```

### Step 1.3: Run Stage 1 (Spatial)

```bash
python sam3/train/train.py \
    -c sam3/train/configs/youtube_vos/ytvos_stage1_spatial.yaml \
    --use-cluster 0 --num-gpus 1
```

### Step 1.4: Run Stage 2 (Temporal)

```bash
# Update config to point to Stage 1 checkpoint
python sam3/train/train.py \
    -c sam3/train/configs/youtube_vos/ytvos_stage2_temporal.yaml \
    --use-cluster 0 --num-gpus 1
```

---

## Phase 2: Factory Domain Adaptation

This is the main section for production factory fine-tuning.

---

## Stage 1: Spatial Adaptation (Images)

### Goal

Teach SAM3 to recognize and segment your factory-specific objects (parts, tools, defects, etc.) using static images with high-quality masks.

### Data Requirements

#### Recommended Dataset Size

| Scenario | Images | Expected Quality |
|----------|--------|------------------|
| Quick PoC | 100-200 | ~70% of optimal |
| Production MVP | 300-500 | ~85% of optimal |
| Robust Production | 500-1000 | ~95% of optimal |

#### Directory Structure

```
factory_dataset/
├── train/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── image_003.jpg
│   └── _annotations.coco.json
└── test/
    ├── image_100.jpg
    ├── image_101.jpg
    └── _annotations.coco.json
```

#### COCO Annotation Format

Your `_annotations.coco.json` should follow this structure:

```json
{
  "info": {
    "description": "Factory Dataset for SAM3 Fine-tuning",
    "version": "1.0",
    "year": 2025
  },
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "width": 1920,
      "height": 1080
    },
    {
      "id": 2,
      "file_name": "image_002.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "categories": [
    {"id": 1, "name": "gripper", "supercategory": "tool"},
    {"id": 2, "name": "part_A", "supercategory": "part"},
    {"id": 3, "name": "defect", "supercategory": "quality"},
    {"id": 4, "name": "screw", "supercategory": "fastener"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 150, 180],
      "area": 27000,
      "segmentation": {
        "counts": "YOUR_RLE_ENCODED_MASK",
        "size": [1080, 1920]
      },
      "iscrowd": 0
    }
  ]
}
```

#### Data Collection Tips

1. **Lighting Diversity**: Capture images under different lighting conditions
2. **Angle Variety**: Include multiple camera angles/viewpoints
3. **Occlusion**: Include partially occluded objects
4. **Background Clutter**: Realistic factory backgrounds
5. **Negative Examples**: 10-20% images where target is NOT present
6. **Per-Class Balance**: Aim for ~100+ examples per object class

### Configuration File

Create `configs/factory/factory_stage1_spatial.yaml`:

```yaml
# @package _global_
# ============================================================================
# Factory Stage 1: Spatial Adaptation (Images with Masks)
# ============================================================================
# Usage:
#   python sam3/train/train.py -c configs/factory/factory_stage1_spatial.yaml \
#       --use-cluster 0 --num-gpus 1
# ============================================================================

defaults:
  - _self_

# ============================================================================
# Paths Configuration (UPDATE THESE!)
# ============================================================================
paths:
  factory_data_root: /path/to/your/factory_dataset
  experiment_log_dir: /path/to/your/experiments/factory_stage1
  bpe_path: sam3/assets/bpe_simple_vocab_16e6.txt.gz

# ============================================================================
# Factory Dataset Configuration
# ============================================================================
factory_train:
  num_images: null  # null = use all images
  
  train_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterCrowds
        - _target_: sam3.train.transforms.point_sampling.RandomizeInputBbox
          box_noise_std: 0.1
          box_noise_max: 20
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes:
            _target_: sam3.train.transforms.basic.get_random_resize_scales
            size: ${scratch.resolution}
            min_size: 480
            rounded: false
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${scratch.resolution}
          square: true
          consistent_transform: ${scratch.consistent_transform}
        - _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
          size: ${scratch.resolution}
          consistent_transform: ${scratch.consistent_transform}
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${scratch.train_norm_mean}
          std: ${scratch.train_norm_std}
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
    - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
      query_filter:
        _target_: sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut
        max_num_objects: ${scratch.max_ann_per_img}

  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes: ${scratch.resolution}
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${scratch.resolution}
          square: true
          consistent_transform: False
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${scratch.train_norm_mean}
          std: ${scratch.train_norm_std}

  # Loss config WITH MASK LOSS
  loss:
    _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
    matcher: ${scratch.matcher}
    o2m_weight: 2.0
    o2m_matcher:
      _target_: sam3.train.matcher.BinaryOneToManyMatcher
      alpha: 0.3
      threshold: 0.4
      topk: 4
    use_o2m_matcher_on_o2m_aux: false
    loss_fns_find:
      - _target_: sam3.train.loss.loss_fns.Boxes
        weight_dict:
          loss_bbox: 5.0
          loss_giou: 2.0
      - _target_: sam3.train.loss.loss_fns.IABCEMdetr
        weak_loss: False
        weight_dict:
          loss_ce: 20.0
          presence_loss: 20.0
        pos_weight: 10.0
        alpha: 0.25
        gamma: 2
        use_presence: True
        pos_focal: false
        pad_n_queries: 200
        pad_scale_pos: 1.0
      # MASK LOSS - Essential for segmentation
      - _target_: sam3.train.loss.loss_fns.Masks
        focal_alpha: 0.25
        focal_gamma: 2.0
        weight_dict:
          loss_mask: 200.0
          loss_dice: 10.0
        compute_aux: false
    loss_fn_semantic_seg: null
    scale_by_find_batch_size: ${scratch.scale_by_find_batch_size}

# ============================================================================
# Scratch Parameters
# ============================================================================
scratch:
  enable_segmentation: True  # CRITICAL: Enable mask training
  
  d_model: 256
  pos_embed:
    _target_: sam3.model.position_encoding.PositionEmbeddingSine
    num_pos_feats: ${scratch.d_model}
    normalize: true
    scale: null
    temperature: 10000

  use_presence_eval: True
  original_box_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessImage
    max_dets_per_img: -1
    use_original_ids: true
    use_original_sizes_box: true
    use_presence: ${scratch.use_presence_eval}

  matcher:
    _target_: sam3.train.matcher.BinaryHungarianMatcherV2
    focal: true
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2
    stable: False
  scale_by_find_batch_size: True

  resolution: 1008
  consistent_transform: False
  max_ann_per_img: 200

  train_norm_mean: [0.5, 0.5, 0.5]
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]

  num_train_workers: 4
  num_val_workers: 0
  max_data_epochs: 40
  target_epoch_size: 1500
  hybrid_repeats: 1
  context_length: 2
  gather_pred_via_filesys: false

  # Differential Learning Rates
  lr_scale: 0.1
  lr_transformer: ${times:8e-4,${scratch.lr_scale}}       # 8e-5 - Highest
  lr_vision_backbone: ${times:2.5e-4,${scratch.lr_scale}} # 2.5e-5 - Medium
  lr_language_backbone: ${times:5e-5,${scratch.lr_scale}} # 5e-6 - Lowest
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 20
  scheduler_cooldown: 20

  val_batch_size: 1
  train_batch_size: 1
  gradient_accumulation_steps: 1

  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${scratch.hybrid_repeats}
    dict_key: factory
    with_seg_masks: ${scratch.enable_segmentation}

  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${scratch.hybrid_repeats}
    dict_key: all
    with_seg_masks: ${scratch.enable_segmentation}

# ============================================================================
# Trainer Configuration
# ============================================================================
trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: false  # Save checkpoints for Stage 2
  empty_gpu_mem_cache_after_eval: True
  skip_first_val: True
  max_epochs: 40
  accelerator: cuda
  seed_value: 123
  val_epoch_freq: 10
  mode: train
  gradient_accumulation_steps: ${scratch.gradient_accumulation_steps}

  distributed:
    backend: nccl
    find_unused_parameters: True
    gradient_as_bucket_view: True

  loss:
    all: ${factory_train.loss}
    default:
      _target_: sam3.train.loss.sam3_loss.DummyLoss

  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        limit_ids: ${factory_train.num_images}
        transforms: ${factory_train.train_transforms}
        load_segmentation: ${scratch.enable_segmentation}
        max_ann_per_img: 500000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: true
        use_caching: False
        img_folder: ${paths.factory_data_root}/train/
        ann_file: ${paths.factory_data_root}/train/_annotations.coco.json
      shuffle: True
      batch_size: ${scratch.train_batch_size}
      num_workers: ${scratch.num_train_workers}
      pin_memory: True
      drop_last: True
      collate_fn: ${scratch.collate_fn}

    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        load_segmentation: ${scratch.enable_segmentation}
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: 2
          _partial_: true
        img_folder: ${paths.factory_data_root}/test/
        ann_file: ${paths.factory_data_root}/test/_annotations.coco.json
        transforms: ${factory_train.val_transforms}
        max_ann_per_img: 100000
        multiplier: 1
        training: false
      shuffle: False
      batch_size: ${scratch.val_batch_size}
      num_workers: ${scratch.num_val_workers}
      pin_memory: True
      drop_last: False
      collate_fn: ${scratch.collate_fn_val}

  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${paths.bpe_path}
    device: cpus
    eval_mode: false
    enable_segmentation: ${scratch.enable_segmentation}

  meters:
    val:
      factory:
        detection:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: "segm"
          dump_dir: ${launcher.experiment_log_dir}/dumps/factory
          merge_predictions: True
          postprocessor: ${scratch.original_box_postprocessor}
          gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}
          maxdets: 100
          pred_file_evaluators:
            - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
              gt_path: ${paths.factory_data_root}/test/_annotations.coco.json
              tide: False
              iou_type: "segm"

  optim:
    amp:
      enabled: True
      amp_dtype: bfloat16
    optimizer:
      _target_: torch.optim.AdamW
    gradient_clip:
      _target_: sam3.train.optim.optimizer.GradientClipper
      max_norm: 0.1
      norm_type: 2
    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: True
        layer_decay_value: ${scratch.lrd_vision_backbone}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides:
          - pattern: '*pos_embed*'
            value: 1.0
    options:
      lr:
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_transformer}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_vision_backbone}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
          param_names:
            - 'backbone.vision_backbone.*'
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_language_backbone}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
          param_names:
            - 'backbone.language_backbone.*'
      weight_decay:
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: ${scratch.wd}
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.0
          param_names:
            - '*bias*'
          module_cls_names: ['torch.nn.LayerNorm']

  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 10

  logging:
    tensorboard_writer:
      _target_: sam3.train.utils.logger.make_tensorboard_logger
      log_dir: ${launcher.experiment_log_dir}/tensorboard
      flush_secs: 120
      should_log: True
    wandb_writer: null
    log_dir: ${launcher.experiment_log_dir}/logs/factory_stage1
    log_freq: 10

launcher:
  num_nodes: 1
  gpus_per_node: 1
  experiment_log_dir: ${paths.experiment_log_dir}
  multiprocessing_context: forkserver

submitit:
  account: null
  partition: null
  qos: null
  timeout_hour: 72
  use_cluster: False
  cpus_per_task: 4
  port_range: [10000, 65000]
  constraint: null
```

---

## Stage 2: Temporal Adaptation (Video)

### Goal

Teach SAM3 to track your factory objects consistently across video frames, reducing mask drift and flicker.

### Data Requirements

#### Recommended Dataset Size

| Scenario | Video Clips | Frames per Clip |
|----------|-------------|-----------------|
| Minimal | 30-50 | 20-50 frames |
| Recommended | 50-100 | 30-100 frames |
| Robust | 100-200 | 50+ frames |

#### Directory Structure

```
factory_videos/
├── train/
│   ├── video_001/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   ├── video_002/
│   │   └── ...
│   └── annotations.json
└── val/
    ├── video_050/
    │   └── ...
    └── annotations.json
```

#### Video Annotation Format (YTVIS-style)

```json
{
  "info": {
    "description": "Factory Video Dataset",
    "version": "1.0"
  },
  "videos": [
    {
      "id": 1,
      "file_names": [
        "video_001/00000.jpg",
        "video_001/00001.jpg",
        "video_001/00002.jpg"
      ],
      "width": 1920,
      "height": 1080,
      "length": 3
    }
  ],
  "categories": [
    {"id": 1, "name": "gripper"},
    {"id": 2, "name": "part_A"}
  ],
  "annotations": [
    {
      "id": 1,
      "video_id": 1,
      "category_id": 1,
      "segmentations": [
        {"counts": "RLE_FRAME_0", "size": [1080, 1920]},
        {"counts": "RLE_FRAME_1", "size": [1080, 1920]},
        null
      ],
      "bboxes": [
        [100, 200, 150, 180],
        [105, 198, 152, 182],
        null
      ],
      "areas": [27000, 27664, null]
    }
  ]
}
```

**Note**: Use `null` for frames where the object is not visible or not annotated.

### Configuration File

Create `configs/factory/factory_stage2_temporal.yaml`:

```yaml
# @package _global_
# ============================================================================
# Factory Stage 2: Temporal Adaptation (Video Tracking)
# ============================================================================
# Usage:
#   python sam3/train/train.py -c configs/factory/factory_stage2_temporal.yaml \
#       --use-cluster 0 --num-gpus 1
#
# IMPORTANT: Run this AFTER Stage 1 completes!
# ============================================================================

defaults:
  - _self_

paths:
  factory_video_root: /path/to/your/factory_videos
  experiment_log_dir: /path/to/your/experiments/factory_stage2
  bpe_path: sam3/assets/bpe_simple_vocab_16e6.txt.gz
  # CRITICAL: Path to Stage 1 checkpoint
  stage1_checkpoint: /path/to/your/experiments/factory_stage1/checkpoints/checkpoint.pt

factory_video_train:
  num_videos: null
  
  train_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterCrowds
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes:
            _target_: sam3.train.transforms.basic.get_random_resize_scales
            size: ${scratch.resolution}
            min_size: 480
            rounded: false
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${scratch.resolution}
          square: true
          consistent_transform: true  # Consistent across frames
        - _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
          size: ${scratch.resolution}
          consistent_transform: true
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${scratch.train_norm_mean}
          std: ${scratch.train_norm_std}
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
    - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
      query_filter:
        _target_: sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut
        max_num_objects: ${scratch.max_ann_per_img}

  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes: ${scratch.resolution}
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${scratch.resolution}
          square: true
          consistent_transform: true
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${scratch.val_norm_mean}
          std: ${scratch.val_norm_std}

  # Video-specific loss with tracking weights
  loss:
    _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
    matcher: ${scratch.matcher}
    o2m_weight: 2.0
    o2m_matcher:
      _target_: sam3.train.matcher.BinaryOneToManyMatcher
      alpha: 0.3
      threshold: 0.4
      topk: 4
    use_o2m_matcher_on_o2m_aux: false
    normalize_by_stage_num: true  # Normalize across frames
    loss_fns_find:
      - _target_: sam3.train.loss.loss_fns.Boxes
        weight_dict:
          loss_bbox: 5.0
          loss_giou: 2.0
        apply_loss_to_det_queries_in_video_grounding: true
      - _target_: sam3.train.loss.loss_fns.IABCEMdetr
        weak_loss: False
        weight_dict:
          loss_ce: 20.0
          presence_loss: 20.0
        pos_weight: 10.0
        alpha: 0.25
        gamma: 2
        use_presence: True
        pos_focal: false
        pad_n_queries: 200
        pad_scale_pos: 1.0
        # Separate detection vs tracking loss
        use_separate_loss_for_det_and_trk: true
        det_exhaustive_loss_scale_pos: 1.0
        det_exhaustive_loss_scale_neg: 1.0
        trk_loss_scale_pos: 2.0  # Weight tracking higher
        trk_loss_scale_neg: 1.0
        apply_loss_to_det_queries_in_video_grounding: true
      - _target_: sam3.train.loss.loss_fns.Masks
        focal_alpha: 0.25
        focal_gamma: 2.0
        weight_dict:
          loss_mask: 200.0
          loss_dice: 10.0
        compute_aux: false
        apply_loss_to_det_queries_in_video_grounding: true
    loss_fn_semantic_seg: null
    scale_by_find_batch_size: ${scratch.scale_by_find_batch_size}

scratch:
  enable_segmentation: True
  d_model: 256
  pos_embed:
    _target_: sam3.model.position_encoding.PositionEmbeddingSine
    num_pos_feats: ${scratch.d_model}
    normalize: true
    scale: null
    temperature: 10000

  use_presence_eval: True
  vid_mask_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessNullOp

  matcher:
    _target_: sam3.train.matcher.BinaryHungarianMatcherV2
    focal: true
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2
    stable: False
  scale_by_find_batch_size: True

  resolution: 1008
  consistent_transform: true
  max_ann_per_img: 100

  train_norm_mean: [0.5, 0.5, 0.5]
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]

  num_train_workers: 4
  num_val_workers: 0
  max_data_epochs: 20
  hybrid_repeats: 1
  gather_pred_via_filesys: false

  # Video sampling
  num_stages_sample: 4
  stage_stride_min: 1
  stage_stride_max: 5
  max_masklet_num_in_video: 50

  # Lower LR for fine-tuning
  lr_scale: 0.05
  lr_transformer: ${times:8e-4,${scratch.lr_scale}}
  lr_vision_backbone: ${times:2.5e-4,${scratch.lr_scale}}
  lr_language_backbone: ${times:5e-5,${scratch.lr_scale}}
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 10
  scheduler_warmup: 10
  scheduler_cooldown: 10

  val_batch_size: 1
  train_batch_size: 1
  gradient_accumulation_steps: 2

  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${scratch.hybrid_repeats}
    dict_key: factory_video
    with_seg_masks: ${scratch.enable_segmentation}

  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${scratch.hybrid_repeats}
    dict_key: all
    with_seg_masks: ${scratch.enable_segmentation}

trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: false
  empty_gpu_mem_cache_after_eval: True
  skip_first_val: True
  max_epochs: 20
  accelerator: cuda
  seed_value: 123
  val_epoch_freq: 5
  mode: train
  gradient_accumulation_steps: ${scratch.gradient_accumulation_steps}

  distributed:
    backend: nccl
    find_unused_parameters: True
    gradient_as_bucket_view: True

  loss:
    all: ${factory_video_train.loss}
    default:
      _target_: sam3.train.loss.sam3_loss.DummyLoss

  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_video_dataset.VideoGroundingDataset
        limit_ids: ${factory_video_train.num_videos}
        transforms: ${factory_video_train.train_transforms}
        load_segmentation: ${scratch.enable_segmentation}
        max_ann_per_img: 100000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: true
        use_caching: False
        num_stages_sample: ${scratch.num_stages_sample}
        stage_stride_min: ${scratch.stage_stride_min}
        stage_stride_max: ${scratch.stage_stride_max}
        random_reverse_time_axis: true
        max_masklet_num_in_video: ${scratch.max_masklet_num_in_video}
        override_query_is_exhaustive_to_true: true
        img_folder: ${paths.factory_video_root}/train/
        ann_file: ${paths.factory_video_root}/train/annotations.json
      shuffle: True
      batch_size: ${scratch.train_batch_size}
      num_workers: ${scratch.num_train_workers}
      pin_memory: True
      drop_last: True
      collate_fn: ${scratch.collate_fn}

    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_video_dataset.VideoGroundingDataset
        load_segmentation: ${scratch.enable_segmentation}
        img_folder: ${paths.factory_video_root}/val/
        ann_file: ${paths.factory_video_root}/val/annotations.json
        transforms: ${factory_video_train.val_transforms}
        max_ann_per_img: 100000
        multiplier: 1
        training: false
        num_stages_sample: 8
        stage_stride_min: 1
        stage_stride_max: 1
      shuffle: False
      batch_size: ${scratch.val_batch_size}
      num_workers: ${scratch.num_val_workers}
      pin_memory: True
      drop_last: False
      collate_fn: ${scratch.collate_fn_val}

  model:
    _target_: sam3.model_builder.build_sam3_video_model
    bpe_path: ${paths.bpe_path}
    has_presence_token: True
    geo_encoder_use_img_cross_attn: True
    apply_temporal_disambiguation: True

  meters:
    val:
      factory_video:
        pred_file:
          _target_: sam3.eval.ytvis_eval.YTVISResultsWriter
          dump_file: ${launcher.experiment_log_dir}/preds/factory_video_val.json
          postprocessor: ${scratch.vid_mask_postprocessor}
          gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}

  optim:
    amp:
      enabled: True
      amp_dtype: bfloat16
    optimizer:
      _target_: torch.optim.AdamW
    gradient_clip:
      _target_: sam3.train.optim.optimizer.GradientClipper
      max_norm: 0.1
      norm_type: 2
    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: True
        layer_decay_value: ${scratch.lrd_vision_backbone}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides:
          - pattern: '*pos_embed*'
            value: 1.0
    options:
      lr:
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_transformer}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_vision_backbone}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
          param_names:
            - 'backbone.vision_backbone.*'
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_language_backbone}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
          param_names:
            - 'backbone.language_backbone.*'
      weight_decay:
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: ${scratch.wd}
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.0
          param_names:
            - '*bias*'
          module_cls_names: ['torch.nn.LayerNorm']

  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 5
    resume_from: ${paths.stage1_checkpoint}  # Load Stage 1!

  logging:
    tensorboard_writer:
      _target_: sam3.train.utils.logger.make_tensorboard_logger
      log_dir: ${launcher.experiment_log_dir}/tensorboard
      flush_secs: 120
      should_log: True
    wandb_writer: null
    log_dir: ${launcher.experiment_log_dir}/logs/factory_stage2
    log_freq: 10

launcher:
  num_nodes: 1
  gpus_per_node: 1
  experiment_log_dir: ${paths.experiment_log_dir}
  multiprocessing_context: forkserver

submitit:
  account: null
  partition: null
  qos: null
  timeout_hour: 72
  use_cluster: False
  cpus_per_task: 4
  port_range: [10000, 65000]
  constraint: null
```

---

## Data Preparation Scripts

### Script 1: Convert Masks to RLE

```python
#!/usr/bin/env python3
"""
convert_masks_to_rle.py

Convert binary mask images to RLE format for COCO annotations.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert a binary mask to RLE format."""
    # Ensure mask is in Fortran order (column-major) as required by pycocotools
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def process_mask_folder(
    mask_folder: str,
    output_json: str,
    image_folder: str,
    categories: list[dict]
):
    """
    Process a folder of mask images and create COCO annotations.
    
    Expected structure:
        mask_folder/
            image_001_class1.png  # Binary mask for class 1 in image_001
            image_001_class2.png  # Binary mask for class 2 in image_001
            ...
    """
    annotations = []
    images = []
    annotation_id = 1
    
    # Get all image files
    image_files = sorted(Path(image_folder).glob("*.jpg"))
    
    for img_id, img_path in enumerate(image_files, start=1):
        img = Image.open(img_path)
        width, height = img.size
        
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })
        
        # Find corresponding masks
        img_stem = img_path.stem
        for cat in categories:
            mask_path = Path(mask_folder) / f"{img_stem}_{cat['name']}.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert('L')) > 127
                if mask.sum() == 0:
                    continue
                
                rle = binary_mask_to_rle(mask)
                bbox = mask_utils.toBbox(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
                area = float(mask_utils.area(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))))
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": cat["id"],
                    "segmentation": rle,
                    "bbox": bbox.tolist(),
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1
    
    coco_format = {
        "info": {"description": "Factory Dataset", "version": "1.0"},
        "images": images,
        "categories": categories,
        "annotations": annotations
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Created {output_json} with {len(images)} images and {len(annotations)} annotations")


if __name__ == "__main__":
    # Define your categories
    categories = [
        {"id": 1, "name": "gripper", "supercategory": "tool"},
        {"id": 2, "name": "part_A", "supercategory": "part"},
        {"id": 3, "name": "defect", "supercategory": "quality"},
    ]
    
    # Process train set
    process_mask_folder(
        mask_folder="factory_dataset/train_masks/",
        output_json="factory_dataset/train/_annotations.coco.json",
        image_folder="factory_dataset/train/",
        categories=categories
    )
    
    # Process test set
    process_mask_folder(
        mask_folder="factory_dataset/test_masks/",
        output_json="factory_dataset/test/_annotations.coco.json",
        image_folder="factory_dataset/test/",
        categories=categories
    )
```

### Script 2: Extract Video Frames

```python
#!/usr/bin/env python3
"""
extract_video_frames.py

Extract frames from video files and organize for SAM3 training.
"""

import os
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    max_frames: int = None
):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every N-th frame
        max_frames: Maximum number of frames to extract
    """
    video_name = Path(video_path).stem
    frame_dir = Path(output_dir) / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = frame_dir / f"{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {frame_dir}")
    return saved_count


def process_video_folder(
    video_folder: str,
    output_dir: str,
    frame_interval: int = 1,
    max_frames_per_video: int = 100
):
    """Process all videos in a folder."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for video_path in Path(video_folder).iterdir():
        if video_path.suffix.lower() in video_extensions:
            extract_frames(
                str(video_path),
                output_dir,
                frame_interval,
                max_frames_per_video
            )


if __name__ == "__main__":
    # Extract training videos
    process_video_folder(
        video_folder="raw_videos/train/",
        output_dir="factory_videos/train/",
        frame_interval=3,  # Every 3rd frame
        max_frames_per_video=100
    )
    
    # Extract validation videos
    process_video_folder(
        video_folder="raw_videos/val/",
        output_dir="factory_videos/val/",
        frame_interval=3,
        max_frames_per_video=50
    )
```

### Script 3: Create Video Annotations (YTVIS Format)

```python
#!/usr/bin/env python3
"""
create_video_annotations.py

Create YTVIS-style annotations for video dataset.
Assumes you have sparse keyframe annotations.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def create_ytvis_annotations(
    video_dir: str,
    keyframe_annotations: dict,  # {video_name: {frame_idx: [annotations]}}
    categories: list[dict],
    output_json: str
):
    """
    Create YTVIS-format annotations from keyframe annotations.
    
    Args:
        video_dir: Directory containing video frame folders
        keyframe_annotations: Sparse annotations for keyframes
        categories: List of category dictionaries
        output_json: Output path for annotations
    """
    videos = []
    annotations = []
    video_id = 1
    annotation_id = 1
    
    for video_folder in sorted(Path(video_dir).iterdir()):
        if not video_folder.is_dir():
            continue
        
        video_name = video_folder.name
        frame_files = sorted(video_folder.glob("*.jpg"))
        
        if not frame_files:
            continue
        
        # Get video dimensions from first frame
        first_frame = Image.open(frame_files[0])
        width, height = first_frame.size
        
        # Create video entry
        videos.append({
            "id": video_id,
            "file_names": [f"{video_name}/{f.name}" for f in frame_files],
            "width": width,
            "height": height,
            "length": len(frame_files)
        })
        
        # Get keyframe annotations for this video
        if video_name in keyframe_annotations:
            video_anns = keyframe_annotations[video_name]
            
            # Group annotations by object_id (track)
            tracks = {}
            for frame_idx, frame_anns in video_anns.items():
                for ann in frame_anns:
                    obj_id = ann.get("object_id", annotation_id)
                    if obj_id not in tracks:
                        tracks[obj_id] = {
                            "category_id": ann["category_id"],
                            "frames": {}
                        }
                    tracks[obj_id]["frames"][frame_idx] = ann
            
            # Create YTVIS annotations for each track
            for obj_id, track in tracks.items():
                segmentations = []
                bboxes = []
                areas = []
                
                for frame_idx in range(len(frame_files)):
                    if frame_idx in track["frames"]:
                        ann = track["frames"][frame_idx]
                        segmentations.append(ann.get("segmentation"))
                        bboxes.append(ann.get("bbox"))
                        areas.append(ann.get("area"))
                    else:
                        # No annotation for this frame
                        segmentations.append(None)
                        bboxes.append(None)
                        areas.append(None)
                
                annotations.append({
                    "id": annotation_id,
                    "video_id": video_id,
                    "category_id": track["category_id"],
                    "segmentations": segmentations,
                    "bboxes": bboxes,
                    "areas": areas
                })
                annotation_id += 1
        
        video_id += 1
    
    ytvis_format = {
        "info": {"description": "Factory Video Dataset", "version": "1.0"},
        "videos": videos,
        "categories": categories,
        "annotations": annotations
    }
    
    with open(output_json, 'w') as f:
        json.dump(ytvis_format, f, indent=2)
    
    print(f"Created {output_json} with {len(videos)} videos and {len(annotations)} tracks")


if __name__ == "__main__":
    # Define categories
    categories = [
        {"id": 1, "name": "gripper"},
        {"id": 2, "name": "part_A"},
    ]
    
    # Example keyframe annotations (you would load these from your annotation tool)
    # Format: {video_name: {frame_idx: [annotation_dicts]}}
    keyframe_annotations = {
        "video_001": {
            0: [
                {
                    "object_id": 1,
                    "category_id": 1,
                    "bbox": [100, 200, 150, 180],
                    "area": 27000,
                    "segmentation": {"counts": "...", "size": [1080, 1920]}
                }
            ],
            10: [
                {
                    "object_id": 1,
                    "category_id": 1,
                    "bbox": [110, 195, 155, 185],
                    "area": 28675,
                    "segmentation": {"counts": "...", "size": [1080, 1920]}
                }
            ]
        }
    }
    
    create_ytvis_annotations(
        video_dir="factory_videos/train/",
        keyframe_annotations=keyframe_annotations,
        categories=categories,
        output_json="factory_videos/train/annotations.json"
    )
```

---

## Training Commands

### Stage 1: Spatial Adaptation

```bash
cd /path/to/sam3_repo

# Single GPU training
python sam3/train/train.py \
    -c configs/factory/factory_stage1_spatial.yaml \
    --use-cluster 0 \
    --num-gpus 1

# Multi-GPU training (if available)
python sam3/train/train.py \
    -c configs/factory/factory_stage1_spatial.yaml \
    --use-cluster 0 \
    --num-gpus 4
```

### Stage 2: Temporal Adaptation

```bash
# AFTER Stage 1 completes
# Make sure to update stage1_checkpoint path in config!

python sam3/train/train.py \
    -c configs/factory/factory_stage2_temporal.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

### SLURM Cluster Training

```bash
# Stage 1 on cluster
python sam3/train/train.py \
    -c configs/factory/factory_stage1_spatial.yaml \
    --use-cluster 1 \
    --partition gpu \
    --account your_account \
    --num-gpus 4 \
    --num-nodes 1

# Stage 2 on cluster
python sam3/train/train.py \
    -c configs/factory/factory_stage2_temporal.yaml \
    --use-cluster 1 \
    --partition gpu \
    --account your_account \
    --num-gpus 4
```

---

## Monitoring & Evaluation

### TensorBoard

```bash
# Monitor training progress
tensorboard --logdir /path/to/experiments/factory_stage1/tensorboard

# In another terminal for Stage 2
tensorboard --logdir /path/to/experiments/factory_stage2/tensorboard
```

### Key Metrics to Watch

| Metric | Stage 1 Target | Stage 2 Target |
|--------|----------------|----------------|
| `loss_bbox` | < 0.5 | < 0.5 |
| `loss_giou` | < 0.5 | < 0.5 |
| `loss_ce` | < 1.0 | < 1.0 |
| `loss_mask` | < 5.0 | < 5.0 |
| `loss_dice` | < 0.5 | < 0.5 |
| Val AP (segm) | > 0.4 | > 0.4 |

### Evaluation Commands

```bash
# Evaluate Stage 1 checkpoint on test set
python sam3/train/train.py \
    -c configs/factory/factory_stage1_spatial.yaml \
    --use-cluster 0 \
    --num-gpus 1
# (Set trainer.mode: val in config)
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```yaml
# Reduce resolution
scratch:
  resolution: 896  # or 768

# Enable gradient accumulation
scratch:
  gradient_accumulation_steps: 4
  train_batch_size: 1
```

#### 2. Training Diverges (Loss goes to NaN)

```yaml
# Reduce learning rate
scratch:
  lr_scale: 0.05  # Half the default

# Increase warmup
scratch:
  scheduler_warmup: 40
```

#### 3. Slow Data Loading

```yaml
# Increase workers (but not too many)
scratch:
  num_train_workers: 8  # Adjust based on CPU cores
```

#### 4. Video Training Crashes

```yaml
# Reduce frames per clip
scratch:
  num_stages_sample: 2  # Start small

# Reduce max objects
scratch:
  max_masklet_num_in_video: 20
```

### Debugging Tips

1. **Start with fewer images**: Test with 50-100 images first
2. **Check data format**: Validate your COCO JSON with `pycocotools`
3. **Monitor GPU memory**: Use `nvidia-smi` during training
4. **Check logs**: Look at `experiment_log_dir/logs/` for errors

---

## Best Practices

### Data Quality

1. **Consistent labeling**: Use the same annotation guidelines throughout
2. **High-quality masks**: Clean boundaries, no holes in solid objects
3. **Diverse conditions**: Vary lighting, angles, backgrounds
4. **Class balance**: Aim for similar counts per class (within 2x)

### Training Strategy

1. **Don't skip Stage 1**: Even if you only care about video, spatial adaptation helps
2. **Validate frequently**: Use `val_epoch_freq: 5` to catch issues early
3. **Save checkpoints**: Keep intermediate checkpoints in case of failures
4. **Monitor loss curves**: Losses should decrease smoothly

### Hyperparameter Tuning

| If you see... | Try... |
|---------------|--------|
| High val loss, low train loss | Increase data, reduce LR, add augmentation |
| Both losses plateau early | Increase LR, increase epochs |
| Unstable training | Reduce LR, increase batch size |
| Slow convergence | Increase LR, reduce warmup |

### Production Deployment

1. **Export final checkpoint**: Use the last checkpoint from Stage 2
2. **Test on held-out data**: Use completely unseen test videos
3. **Benchmark inference speed**: Measure FPS on target hardware
4. **Monitor in production**: Track confidence scores and edge cases

---

## References

- [SAM3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [SAM3 Project Page](https://ai.meta.com/sam3)
- [Roboflow 100 Benchmark](https://github.com/roboflow/roboflow-100-benchmark)
- [COCO Format Specification](https://cocodataset.org/#format-data)
- [YTVIS Format](https://youtube-vos.org/dataset/vis/)

---

## Appendix: Quick Reference

### Config Files Location

```
sam3_repo/sam3/train/configs/factory/
├── factory_stage1_spatial.yaml
└── factory_stage2_temporal.yaml
```

### Key Paths to Update

**Stage 1:**
```yaml
paths:
  factory_data_root: /your/path/here
  experiment_log_dir: /your/path/here
```

**Stage 2:**
```yaml
paths:
  factory_video_root: /your/path/here
  experiment_log_dir: /your/path/here
  stage1_checkpoint: /your/stage1/checkpoint.pt
```

### Expected Training Time

| Stage | Data Size | GPU | Time |
|-------|-----------|-----|------|
| Stage 1 | 500 images | 1x A100 | 4-6 hours |
| Stage 1 | 500 images | 1x 3090 | 8-12 hours |
| Stage 2 | 50 clips | 1x A100 | 6-10 hours |
| Stage 2 | 50 clips | 1x 3090 | 12-18 hours |

