# Reproducing Point Beyond Class (PBC) and Extending with Transformer Backbones

Authors:
- Sundeep Muthukrishnan Kumaraswamy
- Sai Snigdha Nadella

This repository provides a complete reproduction of the Point Beyond Class (PBC) benchmark on RSNA and VinDR-CXR datasets. It extends PBC by analyzing transformer backbones such as ResNet-50, ViT-Base, and Swin-Tiny. This version includes a detailed breakdown of the VinDR-CXR dataset, backbone comparisons, and environment usage for executing Swin and ViT models.

---

## 1. Background & Motivation

Traditional bounding-box annotation for medical images is expensive and slow.  
PBC reduces annotation cost significantly by using **one point per lesion** instead of full bounding boxes.

Key innovations of PBC:
- Point → Box DETR architecture  
- Multi-Point Consistency (MP)  
- Symmetric Consistency (SC)  
- Pseudo-label teacher–student pipeline  

Our project further analyzes how different transformer backbones influence localization performance.

---

## 2. Repository Structure (Detailed)

Point-Beyond-Class/
├── data/  
│   ├── RSNA  
│   ├── cxr  
├── datasets/  
├── models/  
│   ├── detr.py  
│   ├── detr_swin.py  
│   ├── detr_vit.py  
├── student/  
├── outfiles/logs/  
├── pyScripts/  
├── main.py  
├── main_swin.py  
├── main_vit.py  
└── pbc_paper_env.yml  
To run **Swin-Tiny** and **ViT-Base** training, you MUST use the custom environment:

**conda activate pbc_swin_clean**

This environment:
- Contains correct PyTorch + torchvision versions  
- Supports Swin Transformer dependencies  
- Avoids CUDA/torchvision conflicts  
- Ensures compatibility with ViT and DETR backbone code  

If you try running Swin or ViT in another environment, you may face:
- LayerNorm shape mismatches  
- Missing timm dependencies  
- PyTorch/Torchvision compile errors  
- Segmentation faults during training  

For safety:
- **Use pbc_swin_clean for all transformer backbones**
- **Use default PBC environment for ResNet teacher model**

---

## 3. Dataset Preparation — RSNA + VinDR-CXR (Detailed)

### RSNA

RSNA contains 26k+ images with bounding-box pneumonia labels.

RSNA conversion:
cd data/RSNA
python rowimg2jpg.py
python row_csv2tgt_csv.py
python csv2coco_rsna.py

Output:
data/RSNA/cocoAnn/*p/*.json

---

### VinDR-CXR (CXR8)

VinDR-CXR contains:
- 15,000 X-rays  
- 8 key thoracic findings  
- High-quality radiologist annotations  
- More realistic and varied than RSNA  

Why VinDR-CXR is important:
- Higher-quality labels → cleaner supervision  
- Multi-disease → tests backbone generalization  
- Ideal for transformer-based models  

VinDR conversion:
cd data/cxr
python selectDetImgs.py
python generateCsvTxt.py
python csv2coco.py

Output:
data/cxr/ClsAll8_cocoAnnWBF/*p/*.json

---

## 4. PBC Pipeline Breakdown

Stage 1 — Teacher Model  
- Learns bounding boxes from points  
- Variants: Baseline, MP, SC, SC+MP  

Stage 2 — Pseudo Label Generation  
python main.py --eval --resume checkpoint.pth --save_csv pseudo.csv --generate_pseudo_bbox

Stage 3 — Student Model  
python3 tools/train.py configs/faster/faster_xxx.py --seed 42 --deterministic

---

## 5. Teacher Training Commands (Expanded)

Baseline Teacher:
python main.py --epochs 111 --lr_backbone 1e-5 --lr 1e-4 --dataset_file rsna --coco_path data/RSNA/cocoAnn/20p --batch_size 16 --num_workers 16 --data_augment --position_embedding v4 --output_dir outfiles/models/RSNA/exp_baseline

Multi-Point Consistency:
--sample_points_num 2
--cons_loss
--cons_loss_coef 100

Symmetric Consistency:
--sample_points_num 1
--train_with_unlabel_imgs
--unlabel_cons_loss_coef 50
--partial 20

SC + MP:
--load_from checkpoint0110.pth

### NEW: Running Swin and ViT Models (MANDATORY INSTRUCTION)

Before training Swin Transformer or ViT-based DETR:

conda activate pbc_swin_clean

Then run:
python main_swin.py ...
python main_vit.py ...

If this environment is NOT activated:
- Swin forward pass will fail  
- ViT backbone initialization breaks  
- DETR transformer layers mismatch  

---

## 6. Backbone Results VinDR-CXR

### VinDR-CXR (50% partial)

Backbone    | AP50 | Notes
ResNet-50   | 0.2930 | Strong CNN baseline  
ViT-Base    | 0.1982 | Weak for small lesions  
Swin-Tiny   | 0.3069 | Best overall model  

Why Swin wins:
- Hierarchical structure captures multi-scale signals  
- Local window attention suits radiology lesions  
- More stable under consistency losses  



---
## 7. Logs 

All running logs and metrics are present in /Output/outfiles/models/...

## 8. Visualization Scripts

All plots are present in /Output/outfiles/plots and the plots to compare the backbones are present here /Output/outfiles/plots_backbone_compare

---

## 9. Future Work (Extended)

- Design anatomy-aware Swin Transformer for CXR  
- Combine CNN low-level cues + Transformer high-level features  
- Advanced pseudo-label filtering using uncertainty  
- Multi-dataset generalization (NIH, CheXpert, MIMIC-CXR)  
- Explainability pipeline with Grad-CAM, attention rollout  
- Point supervision extended to 3D CT scans  

---

## 10. Citation

@inproceedings{ji2022point,
  title={Point Beyond Class: A Benchmark for Weakly Semi-supervised Abnormality Localization in Chest X-Rays},
  author={Ji, Haoqin and Liu, Haozhe and Li, Yuexiang and Xie, Jinheng and He, Nanjun and Huang, Yawen and Wei, Dong and Chen, Xinrong and Shen, Linlin and Zheng, Yefeng},
  booktitle={MICCAI},
  year={2022}
}

@article{ji2022benchmark,
  title={A Benchmark for Weakly Semi-Supervised Abnormality Localization in Chest X-Rays},
  author={Ji, Haoqin and Liu, Haozhe and Li, Yuexiang and Xie, Jinheng and He, Nanjun and Huang, Yawen and Wei, Dong and Chen, Xinrong and Shen, Linlin and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2209.01988},
  year={2022}
}
