# EdgeAttNet: Towards Barb-Aware Filament Segmentation

Note: This *README* file is for demonstration purposes. For details of the model, please refer to our [paper on arXiv](https://arxiv.org/abs/2509.02964).  
All code files are licensed under the MIT license (see `LICENSE`).



---

[**Paper (coming soon)**]() | [**Dataset (MAGFILO)**](#) 

![EdgeAttNet Architecture](./sample_images/architecture1.png)

Solar filaments are elongated dark structures observed in Hα images, carrying crucial information about solar magnetic fields and space weather events. Detecting them reliably remains challenging due to their fine-scale morphology and imaging artifacts.

We present **EdgeAttNet** — a **U-Net**-based segmentation architecture enhanced with **Edge-Guided Multihead Self-Attention (EG-MHSA)** — designed to accurately segment solar filaments while preserving subtle morphological features such as filament barbs, which are essential for determining the magnetic field orientation of coronal mass ejections (CMEs). By integrating learned edge priors into the attention mechanism, EdgeAttNet outperforms conventional architectures in capturing both global context and fine-grained details, while also reducing the number of trainable parameters compared to U-Net and U-Net+MHSA (with or without positional encodings).


Any questions regarding the model or dataset can be directed to Victor Solomon  (vsolomon3@studentgsu.edu).

## Updates Log
2025.08.12 &emsp; Initial release of EdgeAttNet model and codebase.

## Overview
- **Edge Guided MHSA**: A novel attention mechanism that integrates edge priors into self-attention for better feature extraction.
- **Parameter Efficiency**: Fewer parameters than standard U-Net and MHSA-based models, with faster training and inference.
- **High-resolution Filament Segmentation**: Optimized for capturing detailed filament morphology, enabling downstream analysis such as chirality classification and magnetic field interpretation.

## Methodology

1. **Model Architecture**  
   Built upon the U-Net backbone, EdgeAttNet introduces the **EG-MHSA** module, which leverages edge priors to enhance spatial awareness without explicit positional encodings.

2. **Preprocessing Pipeline**  
   Includes corrections for limb darkening and background interference, ensuring robust performance across diverse solar observations.

3. **Training & Evaluation**  
   Trained and evaluated on the **MAGFILO** dataset, outperforming baseline models including:
   - Standard U-Net
   - U-Net + MHSA (with and without positional encodings)

## 📊 Results

**EdgeAttNet** achieves superior segmentation performance while significantly reducing model complexity:

### 🔧 Model Complexity

| Model                          | Trainable Parameters |
|--------------------------------|----------------------|
| U-Net                          | 31,030,593           |
| U-Net + MHSA (no PE)           | 35,231,041           |
| U-Net + MHSA (with PE)         | 35,362,113           |
| **EdgeAttNet (ours)**          | **22,658,891**       |

This reduction in complexity results in improved training and inference efficiency, as well as better generalization.

Additionally, the integration of edge priors removes the need for positional encodings, simplifying the architecture while maintaining robust spatial feature extraction.

### 📈 Segmentation Performance on MAGFILO Test Split

| Metric                        | U-Net   | U-Net + MHSA (no PE) | U-Net + MHSA (with PE) | **EdgeAttNet (ours)** |
|------------------------------|---------|------------------------|--------------------------|------------------------|
| *mIoU*<sub>pairwise</sub>    | 0.5724  | 0.5856                 | 0.6200                   | **0.6451**             |
| *mIoU*<sub>multiscale</sub>  | 0.5848  | 0.6000                 | 0.6601                   | **0.7032**             |

**EdgeAttNet** consistently outperforms all U-Net variants across both pairwise and multiscale *mIoU* metrics, demonstrating its effectiveness in capturing both global context and fine-scale spatial features.


## Applications

- **Solar Filament Segmentation** in Hα images
- **Chirality Classification** for magnetic field interpretation
- Potential use in **solar eruption forecasting** and **space weather analysis**

## Code Base and Dependencies

All code is written in Python (>=3.8). Deep learning models are implemented in PyTorch and trained on GPU clusters.

| File | Description |
|------|-------------|
| `model/edgeattnet_model.py` | Implementation of the EdgeAttNet architecture & training. |
| `data_loader.py` | Data loading and preprocessing utilities.|
| `main.py` | Contains main function to run model.|
| `requirements.txt` | Required dependencies. |




## 📚 References

[1] S. E. Gibson, “Solar prominences: theory and models: Fleshing out the magnetic skeleton,” *Living Reviews in Solar Physics*, vol. 15, no. 1, p. 7, 2018.  
[2] J. Eastwood, R. Nakamura, L. Turc, L. Mejnertsen, and M. Hesse, “The scientific foundations of forecasting magnetospheric space weather,” *Space Science Reviews*, vol. 212, pp. 1221–1252, 2017.  
[3] S. F. Martin, “Conditions for the formation and maintenance of filaments (invited review),” *Solar Physics*, vol. 182, no. 1, pp. 107–137, 1998.  
[4] ——, “Filament chirality: A link between fine-scale and global patterns,” in *International Astronomical Union Colloquium*, vol. 167. Cambridge University Press, 1998, pp. 419–429.  
[5] Q. Hao, Y. Guo, C. Fang, P.-F. Chen, and W.-D. Cao, “Can we determine the filament chirality by the filament footpoint location or the barb bearing?” *Research in Astronomy and Astrophysics*, vol. 16, no. 1, p. 001, 2016.  
[6] A. Ahmadzadeh, S. S. Mahajan, D. J. Kempton, R. A. Angryk, and S. Ji, “Toward filament segmentation using deep neural networks,” in *IEEE Big Data*, 2019, pp. 4932–4941.  
[7] G. Zhu, G. Lin, X. Yang, and C. Zeng, “Flat U-Net: An efficient ultralightweight model for solar filament segmentation in full-disk H-alpha images,” *arXiv preprint arXiv:2502.07259*, 2025.  
[8] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in *MICCAI*, 2015, pp. 234–241.  
[9] O. Petit, N. Thome, C. Rambour, L. Themyr, T. Collins, and L. Soler, “U-Net Transformer: Self and cross attention for medical image segmentation,” in *MLMI Workshop, MICCAI*, 2021, pp. 267–276.  
[10] X. Qin, C. Wu, H. Chang, H. Lu, and X. Zhang, “Match Feature U-Net: Dynamic receptive field networks for biomedical image segmentation,” *Symmetry*, vol. 12, no. 8, p. 1230, 2020.  
[11] O. Oktay et al., “Attention U-Net: Learning where to look for the pancreas,” in *MICCAI*, Springer, 2018.  
[12] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “SegFormer: Simple and efficient design for semantic segmentation with transformers,” *NeurIPS*, vol. 34, pp. 12077–12090, 2021.  
[13] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional block attention module,” in *ECCV*, Springer, 2018.  
[14] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in *CVPR*, 2018, pp. 7132–7141.  
[15] A. G. Roy, N. Navab, and C. Wachinger, “Concurrent spatial and channel squeeze & excitation in fully convolutional networks,” in *MICCAI*, Springer, 2018, pp. 421–429.  
[16] T. Ge, S.-Q. Chen, and F. Wei, “EdgeFormer: A parameter-efficient transformer for on-device seq2seq generation,” *arXiv preprint arXiv:2202.07959*, 2022.  
[17] A. Ahmadzadeh et al., “A dataset of manually annotated filaments from H-alpha observations,” *Scientific Data*, vol. 11, no. 1, p. 1031, 2024.  
[18] J. Harvey et al., “The Global Oscillation Network Group (GONG) project,” *Science*, vol. 272, no. 5266, pp. 1284–1286, 1996.  
[19] D. Wang and Y. Shang, “A new active labeling method for deep learning,” in *IJCNN*, IEEE, 2014, pp. 112–119.  
[20] A. Men’shchikov, “Background derivation and image flattening: getimages,” *Astronomy & Astrophysics*, vol. 607, p. A64, 2017.  
[21] A. M. Wink and J. B. Roerdink, “Denoising functional MR images: A comparison of wavelet denoising and Gaussian smoothing,” *IEEE Trans. Med. Imaging*, vol. 23, no. 3, pp. 374–387, 2004.  
[22] A. M. Reza, “Realization of the contrast limited adaptive histogram equalization (CLAHE) for real-time image enhancement,” *J. VLSI Signal Processing*, vol. 38, no. 1, pp. 35–44, 2004.  
[23] R. O. Duda and P. E. Hart, “Use of the Hough transformation to detect lines and curves in pictures,” *Communications of the ACM*, vol. 15, no. 1, pp. 11–15, 1972.  
[24] X. Guo et al., “Solar-filament detection and classification based on deep learning,” *Solar Physics*, vol. 297, no. 8, p. 104, 2022.  
[25] A. Diercke et al., “A universal method for solar filament detection from Hα observations using semi-supervised deep learning,” *Astronomy & Astrophysics*, vol. 686, p. A213, 2024.  
[26] H. Ji et al., “A systematic magnetic polarity inversion line dataset from SDO/HMI magnetograms,” *The Astrophysical Journal Supplement Series*, vol. 265, no. 2, p. 40, 2023.  
[27] T.-Y. Lin et al., “Microsoft COCO: Common objects in context,” in *ECCV*, Springer, 2014, pp. 740–755.  
[28] H. Rezatofighi et al., “Generalized intersection over union: A metric and a loss for bounding box regression,” in *CVPR*, 2019, pp. 658–666.  
[29] A. Ahmadzadeh, D. J. Kempton, Y. Chen, and R. A. Angryk, “Multiscale IoU: A metric for evaluation of salient object detection with fine structures,” in *IEEE ICIP*, 2021, pp. 684–688.  

## To install dependencies:

```bash
pip install -r requirements.txt
![alt text](architecture1.png)