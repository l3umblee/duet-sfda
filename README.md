# DUET: Dual-Perspective Pseudo Labeling and Uncertainty-Aware Exploration & Exploitation Training for Source-Free Domain Adaptation

This repository provides the official implementation of **DUET**, accepted as a **NeurIPS 2025 Poster**.

DUET addresses **source-free domain adaptation (SFDA)** by combining:
- **Dual-perspective pseudo labeling**, which assigns pseudo labels only when the target model and CLIP agree.
- **PLMatch**, which trains the target model using pseudo supervision, consistency training, and CLIP-guided knowledge distillation.
- **Uncertainty-aware exploration & exploitation training**, which optimizes the CLIP vision encoder using Tsallis mutual information (TMI).

This codebase is based on [tntek/source-free-domain-adaptation](https://github.com/tntek/source-free-domain-adaptation) and has been modified and extended for our experiments.

---

## 📌 Important Note on Notation

In the graduate thesis version of this work, the KL-divergence terms in Eq. (10) and Eq. (11) contain a typographical sign error.  
The KL-divergence losses should be minimized **without** a leading negative sign.

The released implementation in this repository follows the correct objective and minimizes the KL-divergence terms for consistency training and CLIP-guided alignment.

---

## 📁 Project Structure

This repository includes:
- Source model training scripts
- Target domain adaptation scripts
- Configuration files for Office-Home, VisDA-C, and DomainNet-126
- Utilities for visualization and analysis

---

## 🚀 Source Training

To train a source model for datasets such as **Office-Home**, **VisDA-C**, and **DomainNet-126**, run:

```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/office-home/source.yaml" SETTING.S 0
```

For **DomainNet-126**, we follow the AdaContrast protocol.

---

## 🎯 Target Adaptation

After training the source model, modify `conf.py` to set `${CKPT_DIR}` to the directory containing the source checkpoint.

### Office-Home and VisDA-C

```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/office-home/plmatch.yaml" SETTING.S 0 SETTING.T 1
```

### DomainNet-126

```bash
CUDA_VISIBLE_DEVICES=0 python image_target_in_126.py --cfg "cfgs/domainnet126/plmatch.yaml" SETTING.S 0 SETTING.T 1
```

---

## 📦 Pre-trained Models

Pre-trained source model weights are provided separately and can be used for direct evaluation or target adaptation:

- [Pre-trained source models](https://drive.google.com/drive/folders/17n6goPXw_-ERgTK8R8nm4M_8PJPTEK1j)

---

## 📚 Dataset Preparation

Download the datasets manually and place them in the `./data/` directory.  
You should also prepare the class-name files and domain list files required by each dataset.

An example directory structure is shown below:

```text
data/
├── office-home/
│   ├── Art/
│   ├── Clipart/
│   ├── Product/
│   └── RealWorld/
│
├── VISDA-C/
│   ├── test/
│   ├── train/
│   └── validation/
│
└── domainnet126/
    ├── clipart/
    ├── painting/
    ├── real/
    └── sketch/
```

For **ImageNet variations**, set the `${DATA_DIR}` path in `conf.py` to the corresponding dataset directory.

---

## 🔗 Dataset Download Links

- [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
- [DomainNet-126](https://ai.bu.edu/M3SDA/)

---

## 📝 Notes

- All models were trained and evaluated using a single GPU.
- Configuration files for each dataset are located in the `cfgs/` directory.
- For ImageNet variations, pretrained models from [Torchvision](https://pytorch.org/vision/stable/models.html) or [timm](https://github.com/huggingface/pytorch-image-models) can be used.
- The thesis notation issue mentioned above does not affect the released code or the reported experimental results.

---

## 📧 Contact

For questions or issues, please use the GitHub issue tracker or contact the authors of the NeurIPS submission.
