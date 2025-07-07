# 3D SA-UNet

**3D SA-UNet: 3D Spatial Attention UNet with 3D Atrous Spatial Pyramid Pooling for White Matter Hyperintensities Segmentation**

White Matter Hyperintensity (WMH) is a common imaging biomarker of cerebral small vessel disease and is associated with conditions such as stroke, aging, and dementia. Accurate and automatic WMH segmentation from FLAIR MRI is critical for clinical diagnosis, yet remains challenging due to small lesion size, low contrast, and high discontinuity.

We propose **3D SA-UNet**, a novel 3D convolutional neural network designed specifically for WMH segmentation. It integrates a **3D Spatial Attention Module (3D SAM)** and an enhanced **3D Atrous Spatial Pyramid Pooling (3D ASPP)** to improve feature representation and multi-scale context aggregation.

> 🔬 This project is based on the paper:  
> **Guo, Changlu.** *3D SA-UNet: 3D Spatial Attention UNet with 3D ASPP for White Matter Hyperintensities Segmentation*, arXiv preprint arXiv:2309.08402.

---

## 🧠 Highlights

- **3D Spatial Attention Module (3D SAM):** Enhances the network's ability to focus on lesion regions while suppressing irrelevant background signals.
- **3D Atrous Spatial Pyramid Pooling (3D ASPP):** Captures multi-scale contextual features using 3D dilated convolutions with different dilation rates.
- **Anisotropic Convolutions:** Uses 3×3×1 kernels in encoder/decoder to better handle low z-resolution in MRI.
- **Group Normalization:** Ensures stable training even with small batch sizes.
- **Modular Architecture:** Follows encoder-decoder design for extensibility.

---

## 📐 Architecture

### 3D Spatial Attention Module (3D SAM)

This module applies both average and max pooling along the channel axis and uses a 3D convolution + sigmoid to generate a spatial attention map, which modulates the input features:

<p align="center">
  <img src="3DSAM.png" width="600"/>
  <br/>
  <i>Figure: 3D Spatial Attention Module</i>
</p>

### 3D Atrous Spatial Pyramid Pooling (3D ASPP)

3D ASPP applies parallel 3D convolutions with different dilation rates to learn multi-scale context from 3D volumes:

<p align="center">
  <img src="3daspp.png" width="600"/>
  <br/>
  <i>Figure: 3D ASPP Module</i>
</p>

### Full Model: 3D SA-UNet

The overall architecture consists of encoder–decoder paths with 3D SAM in skip connections and 3D ASPP at the bottleneck:

<p align="center">
  <img src="3dsaunet.png" width="700"/>
  <br/>
  <i>Figure: Full architecture of 3D SA-UNet</i>
</p>

---

## 🧪 Evaluation Protocol

To evaluate the model predictions on WMH segmentation, we recommend using the official script provided by the [WMH Segmentation Challenge (MICCAI 2017)](https://wmh.isi.uu.nl/):

👉 [Official Evaluation Code](https://github.com/hjkuijf/wmhchallenge/blob/master/evaluation.py)

---

## 📂 Dataset

We use the publicly available dataset from the **WMH Segmentation Challenge**, which includes scans from:

- **Utrecht**
- **Singapore**
- **Amsterdam** (GE3T, GE1.5T, Philips 3T)

> Dataset download: [https://wmh.isi.uu.nl/](https://wmh.isi.uu.nl/)

---

## 📊 Results

The table below compares our proposed **3D SA-UNet** with several state-of-the-art methods on WMH segmentation. † indicates results reported from original papers.

| **Model**                         | **DICE ↑** | **AVD ↓** | **F1 Score ↑** |
|----------------------------------|------------|-----------|----------------|
| 3D U-Net       | 0.71       | 0.289     | 0.53           |
| Attention U-Net| 0.74       | 0.206     | 0.57           |
| MICCAI Challenge Winner†  | **0.80**  | 0.219     | **0.76**       |
| **3D SA-UNet (Ours)**            | 0.79       | **0.174** | **0.76**       |

---

## 📄 Citation

If you find this work helpful or use our code/models in your research, please consider citing the following paper:

```bibtex
@article{guo20233d,
  title     = {3D SA-UNet: 3D Spatial Attention UNet with 3D ASPP for White Matter Hyperintensities Segmentation},
  author    = {Guo, Changlu},
  journal   = {arXiv preprint arXiv:2309.08402},
  year      = {2023},
  url       = {https://arxiv.org/abs/2309.08402}
}
