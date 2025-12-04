# Reproducing Point Beyond Class (PBC) and Extending with Transformer Backbones

> **Project:** Reproducing the Point Beyond Class (PBC) benchmark and studying backbone design for weakly semi-supervised abnormality localization in chest X-rays.  
> **Authors:** Sundeep Muthukrishnan Kumaraswamy, Sai Snigdha Nadella  

This repository is based on the original PBC code from MICCAI 2022 and contains our full reproduction on **RSNA** and **VinDR-CXR** datasets, along with an extended **backbone study** comparing **ResNet-50**, **ViT-Base**, and **Swin-Tiny**.

---

## 1. Background & Motivation

Annotating medical images with bounding boxes is:

- Expensive and time-consuming  
- Hard to scale across large datasets  
- Prone to annotation variance  

PBC proposes a scalable alternative: **one point per lesion** instead of a bounding box.

Key ideas behind PBC:

- **Point → Box DETR** model  
- **Multi-Point Consistency (MP)**  
- **Symmetric Consistency (SC)**  
- **Pseudo-label generation** for student training  

Our project goals:

1. Reproduce the original PBC teacher model results  
2. Extend PBC by evaluating **transformer backbones**  
3. Identify which backbone is best for chest X-ray lesion localization  

---

## 2. Repository Layout

```text
Point-Beyond-Class/
├── data/
│   ├── RSNA/
│   │   ├── RSNA_jpg/
│   │   ├── cocoAnn/
│   │   ├── gt_and_pseudo/
│   │   ├── csv2coco_rsna.py
│   │   ├── eval2train_RSNA.py
│   │   └── ...
│   ├── cxr/
│   │   ├── VinBigDataTrain_jpg/
│   │   ├── ClsAll8_cocoAnnWBF/
│   │   ├── gt_and_pseudo/
│   │   ├── csv2coco.py
│   │   ├── eval2train_CXR8.py
│   │   └── ...
│   └── ...
├── datasets/
├── models/
│   ├── detr.py
│   ├── detr_swin.py
│   ├── detr_vit.py
│   ├── backbone.py
│   ├── backbone_swin.py
│   └── ...
├── student/
│   ├── configs/
│   ├── tools/
│   ├── start_CXR8.sh
│   ├── start_RSNA.sh
│   └── ...
├── outfiles/logs/
├── pyScripts/
│   ├── drawLogCXR8.py
│   └── drawLogRSNA.py
├── main.py
├── main_vit.py
├── main_swin.py
├── start_RSNA.sh
├── start_CXR8.sh
└── pbc_paper_env.yml
