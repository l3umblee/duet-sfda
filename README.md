# DUET: Dual-Facet Pseudo Labeling and Uncertainty-aware Exploration & Exploitation Training for Source-Free Adaptation

This repository contains the official implementation for **NIPS 2025 submission** on source-free domain adaptation (SFDA).  
The code is based on the GitHub repository [tntek/source-free-domain-adaptation](https://github.com/tntek/source-free-domain-adaptation) and has been modified and extended for our experimental purposes.

---

## 📁 Project Structure

This repository includes:
- Source model training scripts
- Target domain adaptation scripts
- Configuration files for different datasets
- Utilities for visualization and analysis

---

## 🚀 Source Training

To train a source model for datasets such as **Office-Home**, **VISDA-C**, and **DomainNet-126**, run:

```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/office-home/source.yaml" SETTING.S 0
```

For **DomainNet126**, we follow the AdaContrast protocol.

---

## 🎯 Target Adaptation

After training the source model and modifying `conf.py` to set the `${CKPT_DIR}`, use the following scripts for target adaptation:

### For Office-Home, and VISDA-C:
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/office-home/plmatch.yaml" SETTING.S 0 SETTING.T 1
```

### For DomainNet126:
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_in_126.py --cfg "cfgs/domainnet126/plmatch.yaml" SETTING.S 0 SETTING.T 1
```

---

## 📦 Pre-trained Models

Pre-trained source model weights are **provided separately** and can be used for direct evaluation or adaptation.

---

## 📚 Dataset Preparation

Download the datasets manually and place them in the `./data/` directory. You must also prepare the class name files and list files for each domain.  
An example directory structure for **Office-Home** would look like:

```
data/
├── office-home/
│   ├── Art
│   ├── Clipart
│   ├── Product
│   ├── RealWorld
│
├── VISDA-C/
│   ├── test
│   ├── train
│   ├── validation
│
├── domainnet126/
│   ├── clipart
│   ├── painting
│   ├── real
│   ├── sketch
...
```

For **ImageNet variations**, set the `${DATA_DIR}` path in `conf.py` to your dataset directory accordingly.

---

## 🔗 Dataset Download Links

- **Office-Home**: [Google Drive](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- **VisDA-C**: [GitHub - taskcv-2017-public](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
- **DomainNet-126**: [M3SDA Dataset Page](https://ai.bu.edu/M3SDA/)

---

## 📌 Notes

- All models were trained and evaluated using a single GPU.
- Config files for each dataset are located in the `cfgs/` directory.
- For ImageNet variations, pretrained models from [Torchvision](https://pytorch.org/vision/stable/models.html) or [timm](https://github.com/huggingface/pytorch-image-models) can be used.

---

## 📧 Contact

For any questions or issues, please contact the authors of the NIPS submission.
