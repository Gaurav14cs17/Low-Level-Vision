# Awesome Low-Level Vision — Paper Collection

> A curated collection of papers and code covering low-level and restoration tasks from top computer vision conferences (CVPR, ICCV, ECCV, AAAI, NeurIPS). Updated through **ICCV 2025 / NeurIPS 2025**.

---

## Background

**Low-level tasks** include: Super-Resolution, denoising, deblurring, dehazing, low-light enhancement, artifact removal, etc. The goal is to restore a degraded image into a high-quality one using end-to-end models. Key metrics are PSNR and SSIM. Current challenges include:

*   Poor generalization across datasets — the same model often underperforms on unseen distributions.
*   Gap between objective metrics (PSNR/SSIM) and perceptual/subjective quality.
*   High computational cost of SOTA models (hundreds of GFLOPs) makes real-world deployment difficult.
*   Practical applications include smartphone night modes, beautification filters, security cameras (Hikvision, Dahua), drone imaging (DJI), and video streaming enhancement.

**High-level tasks** include: classification, detection, segmentation, etc. When high-level models receive degraded images, performance drops even with data augmentation. Bridging low-level and high-level vision is an active research direction. Common approaches:

*   Fine-tuning directly on degraded images.
*   Pre-processing with a low-level enhancement network, then passing to the high-level model (two-stage).
*   Joint end-to-end training of the enhancement and high-level task networks.

---

## Table of Contents

> **Legend:** ✅ = Covered &nbsp;|&nbsp; SR = Super-Resolution &nbsp;|&nbsp; DN = Denoising &nbsp;|&nbsp; DB = Deblurring &nbsp;|&nbsp; DR = Deraining &nbsp;|&nbsp; DH = Dehazing &nbsp;|&nbsp; LL = Low-Light &nbsp;|&nbsp; VR = Video Restoration &nbsp;|&nbsp; FI = Frame Interpolation &nbsp;|&nbsp; Gen = Generation/Editing &nbsp;|&nbsp; IQA = Image Quality Assessment

| 📋 Conference | 📅 Year | 📄 Papers | 🔍 SR | 🔇 DN | 💫 DB | 🌧️ DR | 🌫️ DH | 🌙 LL | 🎬 VR | 🎞️ FI | 🎨 Gen | 📊 IQA | ⚙️ Dominant Methods | 🔗 Link |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **CVPR** | **2025** | ~22 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Mamba · Diffusion · MoE · RWKV | [↗ Jump](#cvpr2025-low-level-vision) |
| **ICCV** | **2025** | ~10 | ✅ | ✅ | ✅ | | ✅ | ✅ | | | ✅ | | Diffusion · Autoregressive | [↗ Jump](#iccv2025-low-level-vision) |
| **NeurIPS** | **2025** | ~8 | ✅ | ✅ | ✅ | | | | | | ✅ | | Diffusion · One-Step | [↗ Jump](#neurips2025-low-level-vision) |
| **AAAI** | **2025** | ~5 | ✅ | | | | | | | | | | Diffusion · Transformer | [↗ Jump](#aaai2025-low-level-vision) |
| **CVPR** | **2024** | ~21 | ✅ | ✅ | ✅ | | | ✅ | ✅ | | ✅ | ✅ | Diffusion · Transformer · VLM | [↗ Jump](#cvpr2024-low-level-vision) |
| **ICCV** | **2023** | ~24 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | ✅ | Transformer · Diffusion · Retinex | [↗ Jump](#iccv2023-low-level-vision) |
| **ECCV** | **2024** | ~19 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | Diffusion · Mamba · Transformer | [↗ Jump](#eccv2024-low-level-vision) |
| **NeurIPS** | **2023** | ~11 | ✅ | | | | | ✅ | | | ✅ | ✅ | Diffusion · Consistency Models | [↗ Jump](#neurips2023-low-level-vision) |
| **CVPR** | **2023** | ~110 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Transformer · Diffusion · GAN | [↗ Jump](#cvpr2023-low-level-vision) |
| **CVPR** | **2022** | ~259 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Transformer · GAN · CNN | [↗ Jump](#cvpr2022-low-level-vision) |
| **ECCV** | **2022** | ~256 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Transformer · GAN · Flow | [↗ Jump](#eccv2022-low-level-vision) |
| **AAAI** | **2022** | ~39 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | ✅ | ✅ | CNN · Transformer | [↗ Jump](#aaai2022-low-level-vision) |

> **Total papers indexed: ~784+** across 12 conference tracks (2022–2025)

---

CVPR2025-Low-Level-Vision
=========================

Image Restoration
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration**](https://arxiv.org/abs/2412.20066) | [💻 Code](https://github.com/XLearning-SCU/2025-CVPR-MaIR) | `Mamba, All-in-One, SR/DN/DB/DH` |
| 2 | [**Degradation-Aware Feature Perturbation for All-in-One Image Restoration (DFPIR)**](https://openaccess.thecvf.com/content/CVPR2025/html/Tian_Degradation-Aware_Feature_Perturbation_for_All-in-One_Image_Restoration_CVPR_2025_paper.html) | [💻 Code](https://github.com/TxpHome/DFPIR) | `All-in-One, DN/DH/DR/DB/LL` |
| 3 | [**MoCE-IR: Complexity Experts are Task-Discriminative Learners for Any Image Restoration**](https://arxiv.org/abs/2412.08530) | [💻 Code](https://github.com/eduardzamfir/MoCE-IR) | `Mixture-of-Experts, All-in-One` |
| 4 | [**Visual-Instructed Degradation Diffusion for All-in-One Image Restoration**](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Visual-Instructed_Degradation_Diffusion_for_All-in-One_Image_Restoration_CVPR_2025_paper.pdf) | — | `Diffusion, Visual Instruction, All-in-One` |

### Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution**](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CATANet_Efficient_Content-Aware_Token_Aggregation_for_Lightweight_Image_Super-Resolution_CVPR_2025_paper.html) | — | `Lightweight SR, Transformer, Token Aggregation` |

### Video Restoration

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Hazy Low-Quality Satellite Video Restoration via Learning Optimal Joint Degradation**](https://openaccess.thecvf.com/content/CVPR2025/html/Ni_Hazy_Low-Quality_Satellite_Video_Restoration_Via_Learning_Optimal_Joint_Degradation_CVPR_2025_paper.html) | — | `Video Dehazing, Video SR, Satellite Imagery` |

Deblurring
----------

### Video Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation**](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_EDEN_Enhanced_Diffusion_for_High-quality_Large-motion_Video_Frame_Interpolation_CVPR_2025_paper.html) | — | `Diffusion, Frame Interpolation, Large Motion` |

Image Enhancement
-----------------

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Enhancement**](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_URWKV_Unified_RWKV_Model_with_Multi-state_Perspective_for_Low-light_Image_CVPR_2025_paper.html) | [💻 Code](https://github.com/FZU-N/URWKV) | `RWKV, Low-Light, Joint Deblurring` |
| 2 | [**DarkIR: Robust Low-Light Image Restoration**](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html) | [💻 Code](https://github.com/cidautai/DarkIR) | `CNN, Low-Light, Multi-task` |

Shadow Removal
--------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Detail-Preserving Latent Diffusion for Stable Shadow Removal**](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Detail-Preserving_Latent_Diffusion_for_Stable_Shadow_Removal_CVPR_2025_paper.html) | — | `Diffusion, Shadow Removal, Stable Diffusion Fine-tuning` |

Frame Interpolation
-------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Hierarchical Flow Diffusion for Efficient Frame Interpolation**](https://hfd-interpolation.github.io/) | — | `Diffusion, Frame Interpolation, Optical Flow` |
| 2 | [**Generative Inbetweening through Frame-wise Conditions-Driven Video Generation**](https://fcvg-inbetween.github.io/) | — | `Video Generation, Frame Interpolation, Keyframe` |

Image Matting
-------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**MatAnyone: Stable Video Matting with Consistent Memory Propagation**](https://arxiv.org/abs/2501.03006) | [💻 Code](https://github.com/pq-yang/MatAnyone) | `Video Matting, Memory Propagation, Human Matting` |

Image Quality Assessment
------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Toward Generalized Image Quality Assessment: Relaxing the Perfect Reference Quality Assumption**](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Toward_Generalized_Image_Quality_Assessment_Relaxing_the_Perfect_Reference_Quality_CVPR_2025_paper.html) | — | `IQA, Full-Reference, Generalized, Diffusion` |

Style Transfer
--------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**OmniStyle: Filtering High Quality Style Transfer Data at Scale**](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Dong_Retinex-Guided_Histogram_Transformer_for_Mask-Free_Shadow_Removal_CVPRW_2025_paper.html) | — | `Style Transfer, Diffusion Transformer, Large-Scale Dataset` |

Image Generation/Synthesis
--------------------------

### Video Generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Tora: Trajectory-oriented Diffusion Transformer for Video Generation**](https://arxiv.org/abs/2407.21705) | [💻 Code](https://github.com/alibaba/Tora) | `DiT, Video Generation, Trajectory Control` |
| 2 | [**Mask2DiT: Dual Mask-based Diffusion Transformer for Multi-Scene Long Video Generation**](https://openaccess.thecvf.com/content/CVPR2025/html/Qi_Mask2DiT_Dual_Mask-based_Diffusion_Transformer_for_Multi-Scene_Long_Video_Generation_CVPR_2025_paper.html) | — | `DiT, Long Video, Multi-Scene` |
| 3 | [**Mimir: Improving Video Diffusion Models for Precise Text Understanding**](https://openaccess.thecvf.com/content/CVPR2025/html/Tan_Mimir_Improving_Video_Diffusion_Models_for_Precise_Text_Understanding_CVPR_2025_paper.html) | — | `Video Diffusion, LLM, Text-to-Video` |

### Efficient Generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compute**](https://openaccess.thecvf.com/content/CVPR2025/html/Anagnostidis_FlexiDiT_Your_Diffusion_Transformer_Can_Easily_Generate_High-Quality_Samples_with_CVPR_2025_paper.html) | — | `DiT, Efficient Inference, Adaptive Compute` |

---

ICCV2025-Low-Level-Vision
=========================

Image Restoration
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**UniRes: Universal Image Restoration for Complex Degradations**](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_UniRes_Universal_Image_Restoration_for_Complex_Degradations_ICCV_2025_paper.html) | — | `Diffusion, Universal Restoration, Mixed Degradations` |
| 2 | [**LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling**](https://openaccess.thecvf.com/content/ICCV2025/html/Li_LD-RPS_Zero-Shot_Unified_Image_Restoration_via_Latent_Diffusion_Recurrent_Posterior_ICCV_2025_paper.html) | — | `Zero-Shot, Diffusion, Recurrent Posterior Sampling` |
| 3 | [**Frequency-Guided Posterior Sampling for Diffusion-Based Image Restoration**](https://openaccess.thecvf.com/content/ICCV2025/html/Thaker_Frequency-Guided_Posterior_Sampling_for_Diffusion-Based_Image_Restoration_ICCV_2025_paper.html) | — | `Diffusion, Frequency Domain, Deblurring, Dehazing` |
| 4 | [**Decouple to Reconstruct: High Quality UHD Restoration**](https://openaccess.thecvf.com/content/ICCV2025/html) | — | `UHD, All-in-One, Feature Disentanglement` |

### Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**LightBSR: Towards Lightweight Blind Super-Resolution via Discriminative Implicit Degradation Representation**](https://arxiv.org/abs/2506.22710) | — | `Blind SR, Lightweight, Knowledge Distillation` |
| 2 | [**PURE: Perceive, Understand and Restore Real-World Image Super-Resolution with Autoregressive Multimodal**](https://arxiv.org/abs/2503.11073) | — | `Real-World SR, Autoregressive, Multimodal, LLM` |

Denoising
---------

### Image Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising**](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_IDF_Iterative_Dynamic_Filtering_Networks_for_Generalizable_Image_Denoising_ICCV_2025_paper.html) | — | `Dynamic Filtering, Generalizable, Compact Model` |

Image Enhancement
-----------------

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Noise-Modeled Diffusion Models for Low-Light Spike Image Restoration**](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Noise-Modeled_Diffusion_Models_for_Low-Light_Spike_Image_Restoration_ICCV_2025_paper.html) | — | `Diffusion, Spike Camera, Low-Light, Noise Modeling` |

---

NeurIPS2025-Low-Level-Vision
=============================

Image Restoration
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**DGSolver: Diffusion Generalist Solver with Universal Posterior Sampling for Image Restoration**](https://openreview.net/forum?id=ghhKZ0NaQN) | — | `Diffusion, High-Order Solver, Universal Posterior Sampling` |
| 2 | [**A Minimalistic Unified Framework for Incremental Learning across Image Restoration Tasks**](https://neurips.cc/virtual/2025/poster/118487) | — | `Incremental Learning, Unified Restoration, Meta-Convolution` |

Deblurring
----------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**DeblurDiff: Real-World Image Deblurring with Generative Diffusion Models**](https://neurips.cc/virtual/2025/poster/117332) | — | `Diffusion, Deblurring, Stable Diffusion, Latent Kernel` |

### Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Text-Aware Real-World Image Super-Resolution via Diffusion Model with Joint Segmentation Decoders**](https://neurips.cc/virtual/2025/poster/115913) | — | `Real-World SR, Diffusion, Text-Aware, Segmentation` |
| 2 | [**D³SR: Unleashing the Power of One-Step Diffusion for Image Super-Resolution via a Large-Scale Diffusion Discriminator**](https://openreview.net/forum?id=0M1gi4P4ka) | — | `One-Step Diffusion, SR, Diffusion Discriminator` |

---

AAAI2025-Low-Level-Vision
==========================

### Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Unsupervised Degradation Representation Aware Transform for Real-World Blind Image Super-Resolution**](https://ojs.aaai.org/index.php/AAAI/article/view/32216) | — | `Blind SR, Unsupervised, Degradation Representation` |
| 2 | [**Effective Diffusion Transformer Architecture for Image Super-Resolution**](https://ojs.aaai.org/index.php/AAAI/article/view/32247) | — | `Diffusion Transformer, SR, Frequency-Adaptive` |
| 3 | [**StructSR: Refuse Spurious Details in Real-World Image Super-Resolution**](https://ojs.aaai.org/index.php/AAAI/article/view/32532) | — | `Real-World SR, Diffusion, Structure-Aware Screening` |

---

CVPR2024-Low-Level-Vision
=========================

Image Restoration
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**SUPIR: Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration in the Wild**](https://arxiv.org/abs/2401.13627) | [💻 Code](https://github.com/Fanghua-Yu/SUPIR) | `Diffusion, All-in-One Restoration` |
| 2 | [**PromptIR: Prompting for All-in-One Blind Image Restoration**](https://arxiv.org/abs/2306.13090) | [💻 Code](https://github.com/va1shn9v/PromptIR) | `All-in-One Restoration, Prompt Learning` |
| 3 | [**Photo-Realistic Image Restoration in the Wild with Controlled Vision-Language Models**](https://arxiv.org/abs/2404.09732) | — | `Vision-Language, Blind Restoration` |
| 4 | [**Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration**](https://arxiv.org/abs/2404.00288) | — | `Transformer, Frequency Domain` |

### Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution**](https://arxiv.org/abs/2311.16518) | [💻 Code](https://github.com/cswry/SeeSR) | `Real-World SR, Diffusion, Semantics` |
| 2 | [**APISR: Anime Production Inspired Real-World Anime Super-Resolution**](https://arxiv.org/abs/2403.01819) | [💻 Code](https://github.com/Kiteretsu77/APISR) | `Anime SR, Real-World SR` |
| 3 | [**AddSR: Accelerating Diffusion-based Blind Super-Resolution with Adversarial Diffusion Distillation**](https://arxiv.org/abs/2404.01717) | — | `Diffusion SR, Distillation` |
| 4 | [**XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution**](https://arxiv.org/abs/2403.05049) | [💻 Code](https://github.com/qyp2000/XPSR) | `Diffusion SR, Cross-Modal` |

### Video Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution**](https://arxiv.org/abs/2407.13165) | — | `Video SR, Text-to-Video, Diffusion` |

Denoising
---------

### Image Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Blind Image Restoration via Fast Diffusion Inversion**](https://arxiv.org/abs/2307.07179) | [💻 Code](https://github.com/hamadichihaoui/BIRD) | `Diffusion, Blind Restoration` |

Deblurring
----------

### Image Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ID-Blau: Image Deblurring by Implicit Diffusion-based reBLurring AUgmentation**](https://arxiv.org/abs/2312.10998) | — | `Diffusion, Deblurring` |
| 2 | [**Blur-aware Spatio-temporal Sparse Transformer for Video Deblurring**](https://arxiv.org/abs/2406.07551) | [💻 Code](https://github.com/huicongzhang/BSSTNet) | `Video Deblurring, Transformer` |

Image Enhancement
-----------------

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Zero-Reference Low-Light Video Enhancement Framework Based on Physical Priors**](https://arxiv.org/abs/2403.15700) | — | `Video Low-Light Enhancement, Physical Priors` |
| 2 | [**LLFormer: Ultra-High-Definition Low-Light Image Enhancement**](https://arxiv.org/abs/2212.11548) | [💻 Code](https://github.com/TaoWangzj/LLFormer) | `Low-Light Enhancement, Ultra-HD` |

Image Inpainting
----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Don't Look into the Dark: Latent Codes for Pluralistic Image Inpainting**](https://arxiv.org/abs/2403.06095) | — | `Inpainting, Latent Diffusion` |
| 2 | [**Towards Unified Scene Text Spotting based on Sequence Generation**](https://arxiv.org/abs/2312.05993) | — | `Text Image Restoration` |

Image Generation/Synthesis
--------------------------

### Text-to-Image / Diffusion Models

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models**](https://arxiv.org/abs/2402.19481) | [💻 Code](https://github.com/mit-han-lab/distrifuser) | `Diffusion, Efficient Inference` |
| 2 | [**InstantID: Zero-shot Identity-Preserving Generation in Seconds**](https://arxiv.org/abs/2401.07519) | [💻 Code](https://github.com/InstantX-Team/InstantID) | `Identity-Preserving, Diffusion` |
| 3 | [**InitNo: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization**](https://arxiv.org/abs/2404.04650) | [💻 Code](https://github.com/xiefan-guo/initno) | `Diffusion, Initial Noise` |
| 4 | [**Instruct-Imagen: Image Generation with Multi-modal Instruction**](https://arxiv.org/abs/2401.01952) | — | `Text-to-Image, Multi-modal Instruction` |

Image Quality Assessment
------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**LIQE: Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective**](https://arxiv.org/abs/2312.10656) | [💻 Code](https://github.com/zwx8981/LIQE) | `IQA, Vision-Language` |

---

ICCV2023-Low-Level-Vision
=========================

Image Restoration
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Fourmer: An Efficient Global Modeling Paradigm for Image Restoration**](https://arxiv.org/abs/2308.08974) | — | `Transformer, Frequency Domain` |
| 2 | [**Unified Frequency-Assisted Transformer Framework for Beating Jointly Image Degradation Restoration**](https://arxiv.org/abs/2307.08302) | — | `Transformer, Unified Restoration` |
| 3 | [**Efficient and Degradation-Adaptive Network for Real-World Image Super-Resolution**](https://arxiv.org/abs/2203.04962) | [💻 Code](https://github.com/csjliang/DASR) | `Real-World SR, Adaptive` |
| 4 | [**Kernel Prediction Networks for Blind Single Image Super-Resolution**](https://arxiv.org/abs/2309.05057) | — | `Blind SR` |

### Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**SRFormer: Permuted Self-Attention for Single Image Super-Resolution**](https://arxiv.org/abs/2303.09735) | [💻 Code](https://github.com/HVision-NKU/SRFormer) | `Transformer, Permuted Attention` |
| 2 | [**Boosting Single Image Super-Resolution via Partial Channel Shifting**](https://arxiv.org/abs/2307.07931) | — | `Lightweight SR, Channel Shift` |
| 3 | [**Crafting Training Degradation Distribution for the Accuracy-Generalization Tradeoff in Real-World Super-Resolution**](https://arxiv.org/abs/2305.18596) | [💻 Code](https://github.com/greatlog/RealDAN) | `Real-World SR, Degradation Modeling` |

Denoising
---------

### Image Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Score Priors Guided Deep Variational Inference for Unsupervised Real-World Single Image Denoising**](https://arxiv.org/abs/2309.01183) | — | `Unsupervised, Score-Based` |
| 2 | [**Patch-Craft Self-Supervised Training for Correlated Image Denoising**](https://arxiv.org/abs/2211.09919) | — | `Self-Supervised, Correlated Noise` |

Deblurring
----------

### Image Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Multi-scale Residual Low-Pass Filter Network for Image Deblurring**](https://openaccess.thecvf.com/content/ICCV2023/papers/Dong_Multi-scale_Residual_Low-Pass_Filter_Network_for_Image_Deblurring_ICCV_2023_paper.pdf) | — | `Deblurring, Multi-scale` |

Dehazing
--------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**MB-TaylorFormer: Multi-Branch Efficient Transformer Expanded by Taylor Formula for Image Dehazing**](https://arxiv.org/abs/2308.14036) | [💻 Code](https://github.com/FVL2020/MB-TaylorFormer) | `Transformer, Dehazing` |
| 2 | [**Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration**](https://arxiv.org/abs/2308.03995) | — | `All-in-One, Weather Degradation` |

Image Enhancement
-----------------

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement**](https://arxiv.org/abs/2303.06705) | [💻 Code](https://github.com/caiyuanhao1998/Retinexformer) | `Retinex, Transformer, ICCV Best Paper Candidate` |
| 2 | [**ExposureDiffusion: Learning to Expose for Low-light Image Enhancement**](https://arxiv.org/abs/2307.08927) | [💻 Code](https://github.com/wyf0912/ExposureDiffusion) | `Diffusion, Low-Light` |

### HDR / Exposure Fusion

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning a Simple Low-light Image Enhancer from Paired Low-light Instances**](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf) | — | `Paired Learning, Low-Light` |

Video Restoration
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT)**](https://arxiv.org/abs/2206.02146) | [💻 Code](https://github.com/JingyunLiang/RVRT) | `Video Restoration, Deformable Attention` |
| 2 | [**Benchmark Analysis of Various Enhancement Algorithms for Night-time UAV Object Detection**](https://arxiv.org/abs/2310.18903) | — | `Video Enhancement, Benchmark` |

Image Inpainting
----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Inst-Inpaint: Instructing to Remove Objects with Diffusion Models**](https://arxiv.org/abs/2304.03246) | [💻 Code](https://github.com/abyildirim/inst-inpaint) | `Instruction-Based Inpainting, Diffusion` |

Image Generation/Synthesis
--------------------------

### Text-to-Image / Multi-modal

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation**](https://arxiv.org/abs/2302.13848) | [💻 Code](https://github.com/csyxwei/ELITE) | `Custom Generation, Text-to-Image` |
| 2 | [**Masked Diffusion Transformer is a Strong Image Synthesizer**](https://arxiv.org/abs/2303.14389) | [💻 Code](https://github.com/sail-sg/MDT) | `Masked Diffusion, Transformer` |
| 3 | [**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**](https://arxiv.org/abs/2208.12242) | [💻 Code](https://github.com/google/dreambooth) | `Subject-Driven Generation, Fine-Tuning` |
| 4 | [**Prompt-to-Prompt Image Editing with Cross Attention Control**](https://arxiv.org/abs/2208.01626) | [💻 Code](https://github.com/google/prompt-to-prompt) | `Image Editing, Attention Control` |

### Video Generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence**](https://arxiv.org/abs/2306.02334) | [💻 Code](https://github.com/showlab/VideoSwap) | `Video Editing, Subject Swapping` |

Image Quality Assessment
------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ARNIQA: Learning Distortion Manifold for Image Quality Assessment**](https://arxiv.org/abs/2310.14918) | [💻 Code](https://github.com/miccunifi/ARNIQA) | `IQA, Distortion Manifold` |

---

CVPR2023-Low-Level-Vision
=========================

Image Restoration – Image Restoration
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Efficient and Explicit Modeling of Image Hierarchies for Image Restoration**](https://arxiv.org/abs/2303.00748) | [💻 Code](https://github.com/ofsoundof/GRL-Image-Restoration) | `Transformer` |
| 2 | [**Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**](https://arxiv.org/abs/2303.06859) | [💻 Code](https://github.com/lixinustc/Casual-IRDIL) | — |
| 3 | [**Generative Diffusion Prior for Unified Image Restoration and Enhancement**](https://arxiv.org/abs/2304.01247) | [💻 Code](https://github.com/Fayeben/GenerativeDiffusionPrior) | — |
| 4 | [**Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank**](https://arxiv.org/abs/2303.09101) | [💻 Code](https://github.com/Huang-ShiRui/Semi-UIR) | `Underwater Image Restoration` |
| 5 | [**Nighttime Smartphone Reflective Flare Removal Using Optical Center Symmetry Prior**](https://arxiv.org/abs/2303.15046) | [💻 Code](https://github.com/ykdai/BracketFlare) | `Reflective Flare Removal` |

### Image Reconstruction

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Raw Image Reconstruction with Learned Compact Metadata**](https://arxiv.org/abs/2302.12995) | [💻 Code](https://github.com/wyf0912/R2LCM) | — |
| 2 | [**High-resolution image reconstruction with latent diffusion models from human brain activity**](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v2) | [💻 Code](https://github.com/yu-takagi/StableDiffusionReconstruction) | — |
| 3 | [**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**](https://arxiv.org/abs/2303.06885) | — | — |

### Burst Restoration

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Burstormer: Burst Image Restoration and Enhancement Transformer**](https://arxiv.org/abs/2304.01194) | [💻 Code](https://github.com/akshaydudhane16/Burstormer) | — |

### Video Restoration

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Blind Video Deflickering by Neural Filtering with a Flawed Atlas**](https://arxiv.org/abs/2303.08120) | [💻 Code](https://github.com/ChenyangLEI/All-In-One-Deflicker) | `Deflickering` |

Super Resolution – super resolution
-----------------------------------

### Image Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Activating More Pixels in Image Super-Resolution Transformer**](https://arxiv.org/abs/2205.04437) | [💻 Code](https://github.com/XPixelGroup/HAT) | `Transformer` |
| 2 | [**N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution**](https://arxiv.org/abs/2211.11436) | [💻 Code](https://github.com/rami0205/NGramSwin) | — |
| 3 | **Omni Aggregation Networks for Lightweight Image Super-Resolution** | [💻 Code](https://github.com/Francis0625/Omni-SR) | — |
| 4 | [**OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution**](https://arxiv.org/abs/2303.01091) | — | — |
| 5 | [**Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution**](https://arxiv.org/abs/2303.05156) | — | — |
| 6 | [**Super-Resolution Neural Operator**](https://arxiv.org/abs/2303.02584) | [💻 Code](https://github.com/2y7c3/Super-Resolution-Neural-Operator) | — |
| 7 | [**Human Guided Ground-truth Generation for Realistic Image Super-resolution**](https://arxiv.org/abs/2303.13069) | [💻 Code](https://github.com/ChrisDud0257/PosNegGT) | — |
| 8 | [**Implicit Diffusion Models for Continuous Super-Resolution**](https://arxiv.org/abs/2303.16491) | [💻 Code](https://github.com/Ree1s/IDM) | — |
| 9 | **Zero-Shot Dual-Lens Super-Resolution** | [💻 Code](https://github.com/XrKang/ZeDuSR) | — |
| 10 | [**Learning Generative Structure Prior for Blind Text Image Super-resolution**](https://arxiv.org/abs/2303.14726) | [💻 Code](https://github.com/csxmli2016/MARCONet) | `Text SR` |
| 11 | [**Guided Depth Super-Resolution by Deep Anisotropic Diffusion**](https://arxiv.org/abs/2211.11592) | [💻 Code](https://github.com/prs-eth/Diffusion-Super-Resolution) | `Guided Depth SR` |

### Video Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Towards High-Quality and Efficient Video Super-Resolution via Spatial-Temporal Data Overfitting**](https://arxiv.org/abs/2303.08331) | [💻 Code](https://github.com/coulsonlee/STDO-CVPR2023) | — |
| 2 | [**Structured Sparsity Learning for Efficient Video Super-Resolution**](https://github.com/Zj-BinXia/SSL) | [💻 Code](https://arxiv.org/abs/2206.07687) | — |

Image Rescaling – image scaling
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**HyperThumbnail: Real-time 6K Image Rescaling with Rate-distortion Optimization**](https://arxiv.org/abs/2304.01064) | [💻 Code](https://github.com/AbnerVictor/HyperThumbnail) | — |

Denoising – denoising
---------------------

### Image Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Masked Image Training for Generalizable Deep Image Denoising**](https://arxiv.org/abs/2303.13132) | [💻 Code](https://github.com/haoyuc/MaskedDenoising) | — |
| 2 | [**Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising**](https://arxiv.org/abs/2303.14934) | [💻 Code](https://github.com/nagejacob/SpatiallyAdaptiveSSID) | `Self-Supervised` |
| 3 | [**LG-BPN: Local and Global Blind-Patch Network for Self-Supervised Real-World Denoising**](https://arxiv.org/abs/2304.00534) | [💻 Code](https://github.com/Wang-XIaoDingdd/LGBPN) | `Self-Supervised` |
| 4 | [**Real-time Controllable Denoising for Image and Video**](https://arxiv.org/pdf/2303.16425.pdf) | — | — |

Deblurring – Deblurring
-----------------------

### Image Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Structured Kernel Estimation for Photon-Limited Deconvolution**](https://arxiv.org/abs/2303.03472) | [💻 Code](https://github.com/sanghviyashiitb/structured-kernel-cvpr23) | — |
| 2 | [**Blur Interpolation Transformer for Real-World Motion from Blur**](https://arxiv.org/abs/2211.11423) | [💻 Code](https://github.com/zzh-tech/BiT) | — |
| 3 | **Neumann Network with Recursive Kernels for Single Image Defocus Deblurring** | [💻 Code](https://github.com/csZcWu/NRKNet) | — |
| 4 | [**Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring**](https://arxiv.org/abs/2211.12250) | [💻 Code](https://github.com/kkkls/FFTformer) | — |

Deraining – deraining
---------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning A Sparse Transformer Network for Effective Image Deraining**](https://arxiv.org/abs/2303.11950) | [💻 Code](https://github.com/cschenxiang/DRSformer) | — |

Dehazing – to fog
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | **RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors** | [💻 Code](https://github.com/RQ-Wu/RIDCP) | — |
| 2 | [**Curricular Contrastive Regularization for Physics-aware Single Image Dehazing**](https://arxiv.org/abs/2303.14218) | [💻 Code](https://github.com/YuZheng9/C2PNet) | — |
| 3 | [**Video Dehazing via a Multi-Range Temporal Alignment Network with Physical Prior**](https://arxiv.org/abs/2303.09757) | [💻 Code](https://github.com/jiaqixuac/MAP-Net) | — |

HDR Imaging / Multi-Exposure Image Fusion – HDR image generation / multi-exposure image fusion
----------------------------------------------------------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning a Practical SDR-to-HDRTV Up-conversion using New Dataset and Degradation Models**](https://arxiv.org/abs/2303.13031) | [💻 Code](https://github.com/AndreGuo/HDRTVDM) | — |

Frame Interpolation – frame insertion
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation**](https://arxiv.org/abs/2303.00440) | [💻 Code](https://github.com/MCG-NJU/EMA-VFI) | — |
| 2 | [**A Unified Pyramid Recurrent Network for Video Frame Interpolation**](https://arxiv.org/abs/2211.03456) | [💻 Code](https://github.com/srcn-ivl/UPR-Net) | — |
| 3 | [**BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation**](https://arxiv.org/abs/2304.02225) | [💻 Code](https://github.com/JunHeum/BiFormer) | — |
| 4 | **Event-based Video Frame Interpolation with Cross-Modal Asymmetric Bidirectional Motion Fields** | [💻 Code](https://github.com/intelpro/CBMNet) | `Event-based` |
| 5 | **Event-based Blurry Frame Interpolation under Blind Exposure** | [💻 Code](https://github.com/WarranWeng/EBFI-BE) | `Event-based` |
| 6 | [**Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time**](https://arxiv.org/abs/2303.15043) | [💻 Code](https://github.com/shangwei5/VIDUE) | `Frame Interpolation and Deblurring` |

Image Enhancement – ​​image enhancement
---------------------------------------

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | **Learning Semantic-Aware Knowledge Guidance for Low-Light Image Enhancement** | [💻 Code](https://github.com/langmanbusi/Semantic-Aware-Low-Light-Image-Enhancement) | — |
| 2 | [**Visibility Constrained Wide-band Illumination Spectrum Design for Seeing-in-the-Dark**](https://arxiv.org/abs/2303.11642) | [💻 Code](https://github.com/MyNiuuu/VCSD) | `NIR2RGB` |

Image Matting – image matting
-----------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Referring Image Matting**](https://arxiv.org/abs/2206.05149) | [💻 Code](https://github.com/JizhiziLi/RIM) | — |

Shadow Removal – shadow removal
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ShadowDiffusion: When Degradation Prior Meets Diffusion Model for Shadow Removal**](https://arxiv.org/abs/2212.04711) | [💻 Code](https://github.com/GuoLanqing/ShadowDiffusion) | — |

Image Compression – image compression
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger**](https://arxiv.org/abs/2302.14677) | — | — |
| 2 | [**Context-based Trit-Plane Coding for Progressive Image Compression**](https://arxiv.org/abs/2303.05715) | [💻 Code](https://github.com/seungminjeon-github/CTC) | — |
| 3 | [**Learned Image Compression with Mixed Transformer-CNN Architectures**](https://arxiv.org/abs/2303.14978) | [💻 Code](https://github.com/jmliu206/LIC_TCM) | — |

### Video Compression

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Neural Video Compression with Diverse Contexts**](https://github.com/microsoft/DCVC) | [💻 Code](https://arxiv.org/abs/2302.14402) | — |

Image Quality Assessment – ​​image quality assessment
-----------------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Quality-aware Pre-trained Models for Blind Image Quality Assessment**](https://arxiv.org/abs/2303.00521) | — | — |
| 2 | [**Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective**](https://arxiv.org/abs/2303.14968) | [💻 Code](https://github.com/zwx8981/LIQE) | — |
| 3 | [**Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method**](https://arxiv.org/abs/2303.15166) | [💻 Code](https://github.com/Dreemurr-T/BAID) | — |
| 4 | [**Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild**](https://arxiv.org/abs/2304.00451) | — | — |

Style Transfer – style transfer
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Fix the Noise: Disentangling Source Feature for Controllable Domain Translation**](https://arxiv.org/abs/2303.11545) | [💻 Code](https://github.com/LeeDongYeun/FixNoise) | — |
| 2 | [**Neural Preset for Color Style Transfer**](https://arxiv.org/abs/2303.13511) | [💻 Code](https://github.com/ZHKKKe/NeuralPreset) | — |
| 3 | [**CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer**](https://arxiv.org/abs/2303.17867) | — | — |
| 4 | [**StyleGAN Salon: Multi-View Latent Optimization for Pose-Invariant Hairstyle Transfer**](https://arxiv.org/abs/2304.02744) | [🌐 Project](https://stylegan-salon.github.io/) | — |

Image Editing – image editing
-----------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Imagic: Text-Based Real Image Editing with Diffusion Models**](https://arxiv.org/abs/2210.09276) | — | — |
| 2 | [**SINE: SINgle Image Editing with Text-to-Image Diffusion Models**](https://arxiv.org/abs/2212.04489) | [💻 Code](https://github.com/zhang-zx/SINE) | — |
| 3 | [**CoralStyleCLIP: Co-optimized Region and Layer Selection for Image Editing**](https://arxiv.org/abs/2303.05031) | — | — |
| 4 | [**DeltaEdit: Exploring Text-free Training for Text-Driven Image Manipulation**](https://arxiv.org/abs/2303.06285) | [💻 Code](https://arxiv.org/abs/2303.06285) | — |
| 5 | [**SIEDOB: Semantic Image Editing by Disentangling Object and Background**](https://arxiv.org/abs/2303.13062) | [💻 Code](https://github.com/WuyangLuo/SIEDOB) | — |

Image Generation/Synthesis / Image-to-Image Translation – Image Generation/Synthesis/Translation
------------------------------------------------------------------------------------------------

### Text-to-Image / Text Guided / Multi-Modal

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Multi-Concept Customization of Text-to-Image Diffusion**](https://arxiv.org/abs/2212.04488) | [💻 Code](https://github.com/adobe-research/custom-diffusion) | — |
| 2 | [**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**](https://arxiv.org/abs/2301.12959) | [💻 Code](https://github.com/tobran/GALIP) | — |
| 3 | [**Scaling up GANs for Text-to-Image Synthesis**](https://arxiv.org/abs/2303.05511) | [🌐 Project](https://mingukkang.github.io/GigaGAN/) | — |
| 4 | [**MAGVLT: Masked Generative Vision-and-Language Transformer**](https://arxiv.org/abs/2303.12208) | — | — |
| 5 | [**Freestyle Layout-to-Image Synthesis**](https://arxiv.org/abs/2303.14412) | [💻 Code](https://github.com/essunny310/FreestyleNet) | — |
| 6 | [**Variational Distribution Learning for Unsupervised Text-to-Image Generation**](https://arxiv.org/abs/2303.16105) | — | — |
| 7 | [**Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment**](https://arxiv.org/abs/2303.17490) | [🌐 Project](https://sound2scene.github.io/) | — |
| 8 | [**Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation**](https://arxiv.org/abs/2304.01816) | — | — |

### Image-to-Image / Image Guided

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data**](https://arxiv.org/abs/2208.14889) | [💻 Code](https://github.com/KU-CVLAB/LANIT) | — |
| 2 | [**Person Image Synthesis via Denoising Diffusion Model**](https://arxiv.org/abs/2211.12500) | [💻 Code](https://github.com/ankanbhunia/PIDM) | — |
| 3 | [**Picture that Sketch: Photorealistic Image Generation from Abstract Sketches**](https://arxiv.org/abs/2303.11162) | — | — |
| 4 | [**Fine-Grained Face Swapping via Regional GAN ​​Inversion**](https://arxiv.org/abs/2211.14068) | [💻 Code](https://github.com/e4s2022/e4s) | — |
| 5 | [**Masked and Adaptive Transformer for Exemplar Based Image Translation**](https://arxiv.org/abs/2303.17123) | [💻 Code](https://github.com/AiArt-HDU/MATEBIT) | — |
| 6 | [**Zero-shot Generative Model Adaptation via Image-specific Prompt Learning**](https://arxiv.org/abs/2304.03119) | [💻 Code](https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation) | — |

### Others for image generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**AdaptiveMix: Robust Feature Representation via Shrinking Feature Space**](https://arxiv.org/abs/2303.01559) | [💻 Code](https://github.com/WentianZhang-ML/AdaptiveMix) | — |
| 2 | [**MAGE: MASKed Generative Encoder to Unify Representation Learning and Image Synthesis**](https://arxiv.org/abs/2211.09117) | [💻 Code](https://github.com/LTH14/mage) | — |
| 3 | [**Regularized Vector Quantization for Tokenized Image Synthesis**](https://arxiv.org/abs/2303.06424) | — | — |
| 4 | **Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization** | [💻 Code](https://github.com/CrossmodalGroup/DynamicVectorQuantization) | — |
| 5 | **Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation** | [💻 Code](https://github.com/CrossmodalGroup/MaskedVectorQuantization) | — |
| 6 | **Exploring Incompatible Knowledge Transfer in Few-shot Image Generation** | [💻 Code](https://github.com/yunqing-me/RICK) | — |
| 7 | [**Post-training Quantization on Diffusion Models**](https://arxiv.org/abs/2211.15736) | [💻 Code](https://github.com/42Shawn/PTQ4DM) | — |
| 8 | [**LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation**](https://arxiv.org/abs/2303.17189) | [💻 Code](https://github.com/ZGCTroy/LayoutDiffusion) | — |
| 9 | [**DiffCollage: Parallel Generation of Large Content with Diffusion Models**](https://arxiv.org/abs/2303.17076) | [🌐 Project](https://research.nvidia.com/labs/dir/diffcollage/) | — |
| 10 | [**Few-shot Semantic Image Synthesis with Class Affinity Transfer**](https://arxiv.org/abs/2304.02321) | — | — |

### Video Generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**](https://arxiv.org/abs/2303.13744) | [💻 Code](https://github.com/nihaomiao/CVPR2023_LFDM) | — |
| 2 | [**Video Probabilistic Diffusion Models in Projected Latent Space**](https://arxiv.org/abs/2302.07685) | [💻 Code](https://github.com/sihyun-yu/PVDM) | — |
| 3 | [**DPE: Disentanglement of Pose and Expression for General Video Portrait Editing**](https://arxiv.org/abs/2301.06281) | [💻 Code](https://github.com/Carlyx/DPE) | — |
| 4 | [**Decomposed Diffusion Models for High-Quality Video Generation**](https://arxiv.org/abs/2303.08320) | — | — |
| 5 | [**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding**](https://arxiv.org/abs/2212.02802) | [💻 Code](https://github.com/man805/Diffusion-Video-Autoencoders) | — |
| 6 | **MoStGAN: Video Generation with Temporal Motion Styles** | [💻 Code](https://github.com/xiaoqian-shen/MoStGAN) | — |

### Others

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**DC2: Dual-Camera Defocus Control by Learning to Refocus**](https://arxiv.org/abs/2304.03285) | [🌐 Project](https://defocus-control.github.io/) | — |
| 2 | [**Images Speak in Images: A Generalist Painter for In-Context Visual Learning**](https://arxiv.org/abs/2212.02499) | [💻 Code](https://github.com/baaivision/Painter) | — |
| 3 | [**Unifying Layout Generation with a Decoupled Diffusion Model**](https://arxiv.org/abs/2303.05049) | — | — |
| 4 | [**Unsupervised Domain Adaptation with Pixel-level Discriminator for Image-aware Layout Generation**](https://arxiv.org/abs/2303.14377) | — | — |
| 5 | [**PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout**](https://arxiv.org/abs/2303.15937) | [💻 Code](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023) | — |
| 6 | [**LayoutDM: Discrete Diffusion Model for Controllable Layout Generation**](https://arxiv.org/abs/2303.08137) | [💻 Code](https://github.com/CyberAgentAILab/layout-dm) | — |
| 7 | [**Make-A-Story: Visual Memory Conditioned Consistent Story Generation**](https://arxiv.org/abs/2211.13319) | [💻 Code](https://github.com/ubc-vision/Make-A-Story) | — |
| 8 | [**Cross-GAN Auditing: Unsupervised Identification of Attribute Level Similarities and Differences between Pretrained Generative Models**](https://arxiv.org/abs/2303.10774) | [💻 Code](https://github.com/mattolson93/cross_gan_auditing) | — |
| 9 | [**LightPainter: Interactive Portrait Relighting with Freehand Scribble**](https://arxiv.org/abs/2303.12950) | — | `Portrait Relighting` |
| 10 | **Neural Texture Synthesis with Guided Correspondence** | [💻 Code](https://github.com/EliotChenKJ/Guided-Correspondence-Loss) | `Texture Synthesis` |
| 11 | [**CF-Font: Content Fusion for Few-shot Font Generation**](https://arxiv.org/abs/2303.14017) | [💻 Code](https://github.com/wangchi95/CF-Font) | `Font Generation` |
| 12 | [**DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality**](https://arxiv.org/abs/2303.14585) | [💻 Code](https://github.com/yizhiwang96/deepvecfont-v2) | — |
| 13 | [**Handwritten Text Generation from Visual Archetypes**](https://arxiv.org/abs/2303.15269) | — | `Handwriting Generation` |
| 14 | [**Disentangling Writer and Character Styles for Handwriting Generation**](https://arxiv.org/abs/2303.14736) | [💻 Code](https://github.com/dailenson/SDT) | `Handwriting Generation` |
| 15 | [**Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert**](https://arxiv.org/abs/2303.17480) | [💻 Code](https://github.com/Sxjdwang/TalkLip) | — |
| 16 | [**Uncurated Image-Text Datasets: Shedding Light on Demographic Bias**](https://arxiv.org/abs/2304.02828) | [💻 Code](https://github.com/noagarcia/phase) | — |

CVPR2022-Low-Level-Vision
=========================

Image Restoration – Image Restoration
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Restorer: Efficient Transformer for High-Resolution Image Restoration**](https://arxiv.org/abs/2111.09881) | [💻 Code](https://github.com/swz30/Restormer) | `Transformer` |
| 2 | [**Uformer: A General U-Shaped Transformer for Image Restoration**](https://arxiv.org/abs/2106.03106) | [💻 Code](https://github.com/ZhendongWang6/Uformer) | `Transformer` |
| 3 | [**MAXIM: Multi-Axis MLP for Image Processing**](https://arxiv.org/abs/2201.02973) | [💻 Code](https://github.com/google-research/maxim) | `MLP, also do image enhancement` |
| 4 | [**All-In-One Image Restoration for Unknown Corruption**](http://pengxi.me/wp-content/uploads/2022/03/All-In-One-Image-Restoration-for-Unknown-Corruption.pdf) | [💻 Code](https://github.com/XLearning-SCU/2022-CVPR-AirNet) | — |
| 5 | [**Fourier Document Restoration for Robust Document Dewarping and Recognition**](https://arxiv.org/abs/2203.09910) | — | `Document Restoration` |
| 6 | [**Exploring and Evaluating Image Restoration Potential in Dynamic Scenes**](https://arxiv.org/abs/2203.11754) | — | — |
| 7 | [**ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior**](https://arxiv.org/abs/2111.15362v2) | [💻 Code](https://github.com/ozgurkara99/ISNAS-DIP) | `DIP, NAS` |
| 8 | [**Deep Generalized Unfolding Networks for Image Restoration**](https://arxiv.org/abs/2204.13348) | [💻 Code](https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration) | — |
| 9 | [**Attentive Fine-Grained Structured Sparsity for Image Restoration**](https://arxiv.org/abs/2204.12266) | [💻 Code](https://github.com/JungHunOh/SLS_CVPR2022) | — |
| 10 | [**Self-Supervised Deep Image Restoration via Adaptive Stochastic Gradient Langevin Dynamics**](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Self-Supervised_Deep_Image_Restoration_via_Adaptive_Stochastic_Gradient_Langevin_Dynamics_CVPR_2022_paper.html) | — | `Self-Supervised` |
| 11 | [**KNN Local Attention for Image Restoration**](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_KNN_Local_Attention_for_Image_Restoration_CVPR_2022_paper.html) | [💻 Code](https://sites.google.com/view/cvpr22-kit) | — |
| 12 | [**GIQE: Generic Image Quality Enhancement via Nth Order Iterative Degradation**](https://openaccess.thecvf.com/content/CVPR2022/html/Shyam_GIQE_Generic_Image_Quality_Enhancement_via_Nth_Order_Iterative_Degradation_CVPR_2022_paper.html) | — | — |
| 13 | [**TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions**](https://arxiv.org/abs/2111.14813) | [💻 Code](https://github.com/jeya-maria-jose/TransWeather) | `Adverse Weather` |
| 14 | [**Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model**](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.pdf) | [💻 Code](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal) | `Adverse Weather(deraining, desnowing, dehazing)` |
| 15 | [**Rethinking Deep Face Restoration**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Rethinking_Deep_Face_Restoration_CVPR_2022_paper.html) | — | `face` |
| 16 | [**RestoreFormer: High-Quality Blind Face Restoration From Ungraded Key-Value Pairs**](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_RestoreFormer_High-Quality_Blind_Face_Restoration_From_Undegraded_Key-Value_Pairs_CVPR_2022_paper.html) | [💻 Code](https://github.com/wzhouxiff/RestoreFormer) | `face` |
| 17 | [**Blind Face Restoration via Integrating Face Shape and Generative Priors**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Blind_Face_Restoration_via_Integrating_Face_Shape_and_Generative_Priors_CVPR_2022_paper.html) | — | `face` |
| 18 | [**End-to-End Rubbing Restoration Using Generative Adversarial Networks**](https://arxiv.org/abs/2205.03743) | [💻 Code](https://github.com/qingfengtommy/RubbingGAN) | `\[Workshop\], Rubbing Restoration` |
| 19 | [**GenISP: Neural ISP for Low-Light Machine Cognition**](https://arxiv.org/abs/2205.03688) | — | `\[Workshop\], ISP` |

### Burst Restoration

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift**](https://arxiv.org/abs/2203.09294) | [💻 Code](https://github.com/GuoShi28/2StageAlign) | `joint denoising and demosaicking` |
| 2 | [**Burst Image Restoration and Enhancement**](https://arxiv.org/abs/2110.03680) | [💻 Code](https://github.com/akshaydudhane16/BIPNet) | — |

### Video Restoration

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Revisiting Temporal Alignment for Video Restoration**](https://arxiv.org/abs/2111.15288) | [💻 Code](https://github.com/redrock303/Revisiting-Temporal-Alignment-for-Video-Restoration) | — |
| 2 | [**Neural Compression-Based Feature Learning for Video Restoration**](https://arxiv.org/abs/2203.09208) | — | — |
| 3 | [**Bringing Old Films Back to Life**](https://arxiv.org/abs/2203.17276) | [💻 Code](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life) | — |
| 4 | [**Neural Global Shutter: Learn to Restore Video from a Rolling Shutter Camera with Global Reset Feature**](https://arxiv.org/abs/2204.00974) | [💻 Code](https://github.com/lightChaserX/neural-global-shutter) | `restore clean global shutter (GS) videos` |
| 5 | [**Context-Aware Video Reconstruction for Rolling Shutter Cameras**](https://arxiv.org/abs/2205.12912) | [💻 Code](https://github.com/GitCVfb/CVR) | `Rolling Shutter Cameras` |
| 6 | [**E2V-SDE: From Asynchronous Events to Fast and Continuous Video Reconstruction via Neural Stochastic Differential Equations**](https://arxiv.org/abs/2206.07578) | — | `Event camera` |

### Hyperspectral Image Reconstruction

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction**](https://arxiv.org/abs/2111.07910) | [💻 Code](https://github.com/caiyuanhao1998/MST) | — |
| 2 | [**HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging**](https://arxiv.org/abs/2203.02149) | — | — |

Super Resolution – super resolution
-----------------------------------

### Image Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Reflash Dropout in Image Super-Resolution**](https://arxiv.org/abs/2112.12089) | [💻 Code](https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution) | — |
| 2 | [**Residual Local Feature Network for Efficient Super-Resolution**](https://arxiv.org/abs/2205.07514) | [💻 Code](https://github.com/fyan111/RLFN) | `won the first place in the runtime track of the NTIRE 2022 efficient super-resolution challenge` |
| 3 | [**Learning the Degradation Distribution for Blind Image Super-Resolution**](https://arxiv.org/abs/2203.04962) | [💻 Code](https://arxiv.org/abs/2203.04962) | `Blind SR` |
| 4 | [**Deep Constrained Least Squares for Blind Image Super-Resolution**](https://arxiv.org/abs/2202.07508) | [💻 Code](https://github.com/Algolzw/DCLS) | `Blind SR` |
| 5 | [**Blind Image Super-resolution with Elaborate Degradation Modeling on Noise and Kernel**](https://arxiv.org/abs/2107.00986) | [💻 Code](https://github.com/zsyOAOA/BSRDM) | `Blind SR` |
| 6 | [**Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution**](https://arxiv.org/abs/2203.09195) | [💻 Code](https://github.com/csjliang/LDL) | `Real SR` |
| 7 | [**Dual Adversarial Adaptation for Cross-Device Real-World Image Super-Resolution**](https://arxiv.org/abs/2205.03524) | [💻 Code](https://github.com/lonelyhope/DADA) | `Real SR` |
| 8 | [**LAR-SR: A Local Autoregressive Model for Image Super-Resolution**](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_LAR-SR_A_Local_Autoregressive_Model_for_Image_Super-Resolution_CVPR_2022_paper.html) | — | — |
| 9 | [**Texture-Based Error Analysis for Image Super-Resolution**](https://openaccess.thecvf.com/content/CVPR2022/html/Magid_Texture-Based_Error_Analysis_for_Image_Super-Resolution_CVPR_2022_paper.html) | — | — |
| 10 | [**Learning to Zoom Inside Camera Imaging Pipeline**](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Learning_To_Zoom_Inside_Camera_Imaging_Pipeline_CVPR_2022_paper.html) | — | `Raw-to-Raw domain` |
| 11 | [**Task Decoupled Framework for Reference-Based Super-Resolution**](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.html) | — | `Reference-Based` |
| 12 | [**GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors**](https://arxiv.org/abs/2203.07319) | [💻 Code](https://github.com/hejingwenhejingwen/GCFSR) | `Face SR` |
| 13 | [**A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution**](https://arxiv.org/abs/2203.09388) | [💻 Code](https://github.com/mjq11302010044/TATT) | `Text SR` |
| 14 | [**Learning Graph Regularization for Guided Super-Resolution**](https://arxiv.org/abs/2203.14297) | — | `Guided SR` |
| 15 | [**Transformer-empowered Multi-scale Contextual Matching and Aggregation for Multi-contrast MRI Super-resolution**](https://arxiv.org/abs/2203.13963) | [💻 Code](https://github.com/XAIMI-Lab/McMRSR) | `MRI SR` |
| 16 | [**Discrete Cosine Transform Network for Guided Depth Map Super-Resolution**](https://arxiv.org/abs/2104.06977) | [💻 Code](https://github.com/Zhaozixiang1228/GDSR-DCTNet) | `Guided Depth Map SR` |
| 17 | [**SphereSR: 360deg Image Super-Resolution With Arbitrary Projection via Continuous Spherical Image Representation**](https://openaccess.thecvf.com/content/CVPR2022/html/Yoon_SphereSR_360deg_Image_Super-Resolution_With_Arbitrary_Projection_via_Continuous_Spherical_CVPR_2022_paper.html) | — | — |
| 18 | [**IM Deception: Grouped Information Distilling Super-Resolution Network**](https://arxiv.org/abs/2204.11463) | — | `\[Workshop\], lightweight` |
| 19 | [**A Closer Look at Blind Super-Resolution: Degradation Models, Baselines, and Performance Upper Bounds**](https://arxiv.org/abs/2205.04910) | [💻 Code](https://github.com/WenlongZhang0517/CloserLookBlindSR) | `\[Workshop\], Blind SR` |

### Burst/Multi-frame Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites**](https://arxiv.org/abs/2205.02031) | [💻 Code](https://github.com/centreborelli/HDR-DSP-SR/) | `Self-Supervised, multi-exposure` |

### Video Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment**](https://arxiv.org/abs/2104.13371) | [💻 Code](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) | — |
| 2 | [**Learning Trajectory-Aware Transformer for Video Super-Resolution**](https://arxiv.org/abs/2204.04216) | [💻 Code](https://github.com/researchmm/TTVSR) | `Transformer` |
| 3 | [**Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling**](https://arxiv.org/abs/2204.07114) | — | — |
| 4 | [**Investigating Tradeoffs in Real-World Video Super-Resolution**](https://arxiv.org/abs/2111.12704) | [💻 Code](https://github.com/ckkelvinchan/RealBasicVSR) | `Real-world, RealBaiscVSR` |
| 5 | [**Memory-Augmented Non-Local Attention for Video Super-Resolution**](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Memory-Augmented_Non-Local_Attention_for_Video_Super-Resolution_CVPR_2022_paper.html) | — | — |
| 6 | [**Stable Long-Term Recurrent Video Super-Resolution**](https://openaccess.thecvf.com/content/CVPR2022/html/Chiche_Stable_Long-Term_Recurrent_Video_Super-Resolution_CVPR_2022_paper.html) | — | — |
| 7 | [**Reference-based Video Super-Resolution Using Multi-Camera Video Triplets**](https://arxiv.org/abs/2203.14537) | [💻 Code](https://github.com/codeslake/RefVSR) | `Reference-based VSR` |
| 8 | [**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**](https://arxiv.org/abs/2204.10039) | [💻 Code](https://github.com/H-deep/Trans-SVSR/) | `Stereoscopic Video Super-Resolution` |

Image Rescaling – image scaling
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence**](https://arxiv.org/abs/2203.00911) | — | — |
| 2 | [**Faithful Extreme Rescaling via Generative Prior Reciprocated Invertible Representations**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_Faithful_Extreme_Rescaling_via_Generative_Prior_Reciprocated_Invertible_Representations_CVPR_2022_paper.html) | [💻 Code](https://github.com/cszzx/GRAIN) | — |

Denoising – denoising
---------------------

### Image Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Self-Supervised Image Denoising via Iterative Data Refinement**](https://arxiv.org/abs/2111.14358) | [💻 Code](https://github.com/zhangyi-3/IDR) | `Self-Supervised` |
| 2 | [**Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots**](https://arxiv.org/abs/2203.06967) | [💻 Code](https://github.com/demonsjin/Blind2Unblind) | `Self-Supervised` |
| 3 | [**AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network**](https://arxiv.org/abs/2203.11799) | [💻 Code](https://github.com/wooseoklee4/AP-BSN) | `Self-Supervised` |
| 4 | [**CVF-SID: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image**](https://arxiv.org/abs/2203.13009) | [💻 Code](https://github.com/Reyhanehne/CVF-SID_PyTorch) | `Self-Supervised` |
| 5 | [**Noise Distribution Adaptive Self-Supervised Image Denoising Using Tweedie Distribution and Score Matching**](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Noise_Distribution_Adaptive_Self-Supervised_Image_Denoising_Using_Tweedie_Distribution_and_CVPR_2022_paper.html) | — | `Self-Supervised` |
| 6 | [**Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images**](https://arxiv.org/abs/2206.01103) | — | `Noise Modeling, Normalizing Flow` |
| 7 | [**Modeling sRGB Camera Noise with Normalizing Flows**](https://arxiv.org/abs/2206.00812) | — | `Noise Modeling, Normalizing Flow` |
| 8 | [**Estimating Fine-Grained Noise Model via Contrastive Learning**](https://openaccess.thecvf.com/content/CVPR2022/html/Zou_Estimating_Fine-Grained_Noise_Model_via_Contrastive_Learning_CVPR_2022_paper.html) | — | `Noise Modeling, Constrastive Learning` |
| 9 | [**Multiple Degradation and Reconstruction Network for Single Image Denoising via Knowledge Distillation**](https://arxiv.org/abs/2204.13873) | — | `\[Workshop\]` |

### Burst Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**NAN: Noise-Aware NeRFs for Burst-Denoising**](https://arxiv.org/abs/2204.04668) | — | `NeRFs` |

### Video Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Dancing under the stars: video denoising in starlight**](https://arxiv.org/abs/2204.04210) | [💻 Code](https://github.com/monakhova/starlight_denoising/) | `video denoising in starlight` |

Deblurring – Deblurring
-----------------------

### Image Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning to Deblur using Light Field Generated and Real Defocus Images**](https://arxiv.org/abs/2204.00367) | [💻 Code](https://github.com/lingyanruan/DRBNet) | `Defocus deblurring` |
| 2 | [**Pixel Screening Based Intermediate Correction for Blind Deblurring**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Pixel_Screening_Based_Intermediate_Correction_for_Blind_Deblurring_CVPR_2022_paper.html) | — | `Blind` |
| 3 | [**Deblurring via Stochastic Refinement**](https://openaccess.thecvf.com/content/CVPR2022/html/Whang_Deblurring_via_Stochastic_Refinement_CVPR_2022_paper.html) | — | — |
| 4 | [**XYDeblur: Divide and Conquer for Single Image Deblurring**](https://openaccess.thecvf.com/content/CVPR2022/html/Ji_XYDeblur_Divide_and_Conquer_for_Single_Image_Deblurring_CVPR_2022_paper.html) | — | — |
| 5 | [**Unifying Motion Deblurring and Frame Interpolation with Events**](https://arxiv.org/abs/2203.12178) | — | `event-based` |
| 6 | [**E-CIR: Event-Enhanced Continuous Intensity Recovery**](https://arxiv.org/abs/2203.01935) | [💻 Code](https://github.com/chensong1995/E-CIR) | `event-based` |

### Video Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Multi-Scale Memory-Based Video Deblurring**](https://arxiv.org/abs/2203.01935) | [💻 Code](https://github.com/jibo27/MemDeblur) | — |

Deraining – deraining
---------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**](https://arxiv.org/abs/2203.16931) | [💻 Code](https://github.com/yuyi-sd/Robust_Rain_Removal) | — |
| 2 | [**Unpaired Deep Image Deraining Using Dual Contrastive Learning**](https://arxiv.org/abs/2109.02973) | — | `Contrastive Learning, Unpaired` |
| 3 | [**Unsupervised Deraining: Where Contrastive Learning Meets Self-similarity**](https://arxiv.org/abs/2203.11509) | — | `Contrastive Learning, Unsupervised` |
| 4 | [**Dreaming To Prune Image Deraining Networks**](https://openaccess.thecvf.com/content/CVPR2022/html/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.html) | — | — |

Dehazing – to fog
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Self-augmented Unpaired Image Dehazing via Density and Depth Decomposition**](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Self-Augmented_Unpaired_Image_Dehazing_via_Density_and_Depth_Decomposition_CVPR_2022_paper.html) | [💻 Code](https://github.com/YaN9-Y/D4) | `Unpaired` |
| 2 | [**Towards Multi-Domain Single Image Dehazing via Test-Time Training**](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Multi-Domain_Single_Image_Dehazing_via_Test-Time_Training_CVPR_2022_paper.html) | — | — |
| 3 | [**Image Dehazing Transformer With Transmission-Aware 3D Position Embedding**](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Image_Dehazing_Transformer_With_Transmission-Aware_3D_Position_Embedding_CVPR_2022_paper.html) | — | — |
| 4 | [**Physically Disentangled Intra- and Inter-Domain Adaptation for Varicolored Haze Removal**](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Physically_Disentangled_Intra-_and_Inter-Domain_Adaptation_for_Varicolored_Haze_Removal_CVPR_2022_paper.html) | — | — |

Demoireing – Go moiré
---------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Video Demoireing with Relation-Based Temporal Consistency**](https://arxiv.org/abs/2204.02957) | [💻 Code](https://github.com/CVMI-Lab/VideoDemoireing) | — |

Frame Interpolation – frame insertion
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ST-MFNet: A Spatio-Temporal Multi-Flow Network for Frame Interpolation**](https://arxiv.org/abs/2111.15483) | [💻 Code](https://github.com/danielism97/ST-MFNet) | — |
| 2 | [**Long-term Video Frame Interpolation via Feature Propagation**](https://arxiv.org/abs/2203.15427) | — | — |
| 3 | [**Many-to-many Splatting for Efficient Video Frame Interpolation**](https://arxiv.org/abs/2204.03513) | [💻 Code](https://github.com/feinanshan/M2M_VFI) | — |
| 4 | [**Video Frame Interpolation with Transformer**](https://arxiv.org/abs/2205.07230) | [💻 Code](https://github.com/dvlab-research/VFIformer) | `Transformer` |
| 5 | [**Video Frame Interpolation Transformer**](https://arxiv.org/abs/2111.13817) | [💻 Code](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer) | `Transformer` |
| 6 | [**IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation**](https://arxiv.org/abs/2205.14620) | [💻 Code](https://github.com/ltkong218/IFRNet) | — |
| 7 | [**TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation**](https://arxiv.org/abs/2203.13859) | — | `Event Camera` |
| 8 | [**Time Lens++: Event-based Frame Interpolation with Parametric Non-linear Flow and Multi-scale Fusion**](https://arxiv.org/abs/2203.17191) | — | `Event-based` |
| 9 | [**Unifying Motion Deblurring and Frame Interpolation with Events**](https://arxiv.org/abs/2203.12178) | — | `event-based` |
| 10 | [**Multi-encoder Network for Parameter Reduction of a Kernel-based Interpolation Architecture**](https://arxiv.org/abs/2205.06723) | — | `\[Workshop\]` |

### Spatial-Temporal Video Super-Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution**](https://arxiv.org/abs/2203.14186) | [💻 Code](https://github.com/llmpass/RSTT) | — |
| 2 | [**Spatial-Temporal Space Hand-in-Hand: Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning**](https://arxiv.org/abs/2205.05264) | — | — |
| 3 | [**VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution**](https://arxiv.org/abs/2206.04647) | [💻 Code](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution) | — |

Image Enhancement – ​​image enhancement
---------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-time Image Enhancement**](https://arxiv.org/abs/2204.13983) | [💻 Code](https://github.com/ImCharlesY/AdaInt) | — |
| 2 | [**Exposure Correction Model to Enhance Image Quality**](https://arxiv.org/abs/2204.10648) | [💻 Code](https://github.com/yamand16/ExposureCorrection) | `\[Workshop\]` |

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Abandoning the Bayer-Filter to See in the Dark**](https://arxiv.org/abs/2203.04042) | [💻 Code](https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark) | — |
| 2 | [**Toward Fast, Flexible, and Robust Low-Light Image Enhancement**](https://arxiv.org/abs/2204.10137) | [💻 Code](https://github.com/vis-opt-group/SCI) | — |
| 3 | [**Deep Color Consistent Network for Low-Light Image Enhancement**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Deep_Color_Consistent_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html) | — | — |
| 4 | [**SNR-Aware Low-Light Image Enhancement**](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html) | [💻 Code](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance) | — |
| 5 | [**URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement**](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html) | — | — |

Image Harmonization – Image Harmonization
-----------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**High-Resolution Image Harmonization via Collaborative Dual Transformationsg**](https://arxiv.org/abs/2109.06671) | [💻 Code](https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization) | — |
| 2 | [**SCS-Co: Self-Consistent Style Contrastive Learning for Image Harmonization**](https://arxiv.org/abs/2204.13962) | [💻 Code](https://github.com/YCHang686/SCS-Co-CVPR2022) | — |
| 3 | [**Deep Image-based Illumination Harmonization**](https://arxiv.org/abs/2108.00150) | — | — |

Image Completion/Inpainting – image restoration
-----------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Bridging Global Context Interactions for High-Fidelity Image Completion**](https://arxiv.org/abs/2104.00845) | [💻 Code](https://github.com/lyndonzheng/TFill) | — |
| 2 | [**Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding**](https://arxiv.org/abs/2203.00867) | [💻 Code](https://github.com/DQiaole/ZITS_inpainting) | — |
| 3 | [**MISF: Multi-level Interactive Siamese Filtering for High-Fidelity Image Inpainting**](https://arxiv.org/abs/2203.06304) | [💻 Code](https://github.com/tsingqguo/misf) | — |
| 4 | [**MAT: Mask-Aware Transformer for Large Hole Image Inpainting**](https://arxiv.org/abs/2203.15270) | [💻 Code](https://github.com/fenglinglwb/MAT) | — |
| 5 | [**Reduce Information Loss in Transformers for Pluralistic Image Inpainting**](https://arxiv.org/abs/2205.05076) | [💻 Code](https://github.com/liuqk3/PUT) | — |
| 6 | [**RePaint: Inpainting using Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2201.09865) | [💻 Code](https://github.com/andreas128/RePaint) | `DDPM` |
| 7 | [**Dual-Path Image Inpainting With Auxiliary GAN Inversion**](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Dual-Path_Image_Inpainting_With_Auxiliary_GAN_Inversion_CVPR_2022_paper.html) | — | — |
| 8 | [**SaiNet: Stereo aware inpainting behind objects with generative networks**](https://arxiv.org/abs/2205.07014) | — | `\[Workshop\]` |

### Video Inpainting

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Towards An End-to-End Framework for Flow-Guided Video Inpainting**](https://arxiv.org/abs/2204.02663) | [💻 Code](https://github.com/MCG-NKU/E2FGVI) | — |
| 2 | [**The DEVIL Is in the Details: A Diagnostic Evaluation Benchmark for Video Inpainting**](https://openaccess.thecvf.com/content/CVPR2022/html/Szeto_The_DEVIL_Is_in_the_Details_A_Diagnostic_Evaluation_Benchmark_CVPR_2022_paper.html) | [💻 Code](https://github.com/MichiganCOG/devil) | — |
| 3 | [**DLFormer: Discrete Latent Transformer for Video Inpainting**](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_DLFormer_Discrete_Latent_Transformer_for_Video_Inpainting_CVPR_2022_paper.html) | — | — |
| 4 | [**Inertia-Guided Flow Completion and Style Fusion for Video Inpainting**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Inertia-Guided_Flow_Completion_and_Style_Fusion_for_Video_Inpainting_CVPR_2022_paper.htmll) | — | — |

Image Matting – image matting
-----------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**MatteFormer: Transformer-Based Image Matting via Prior-Tokens**](https://arxiv.org/abs/2203.15662) | [💻 Code](https://github.com/webtoon/matteformer) | — |
| 2 | [**Human Instance Matting via Mutual Guidance and Multi-Instance Refinement**](https://arxiv.org/abs/2205.10767) | [💻 Code](https://github.com/nowsyn/InstMatt) | — |
| 3 | [**Boosting Robustness of Image Matting with Context Assembly and Strong Data Augmentation**](https://arxiv.org/abs/2201.06889) | — | — |

Shadow Removal – shadow removal
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Bijective Mapping Network for Shadow Removal**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Bijective_Mapping_Network_for_Shadow_Removal_CVPR_2022_paper.html) | — | — |

Relighting
----------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Face Relighting with Geometrically Consistent Shadows**](https://arxiv.org/abs/2203.16681) | [💻 Code](https://github.com/andrewhou1/GeomConsistentFR) | `Face Relighting` |
| 2 | [**SIMBAR: Single Image-Based Scene Relighting For Effective Data Augmentation For Automated Driving Vision Tasks**](https://arxiv.org/abs/2204.00644) | — | — |

Image Stitching – image stitching
---------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Deep Rectangling for Image Stitching: A Learning Baseline**](https://arxiv.org/abs/2203.03831) | [💻 Code](https://github.com/nie-lang/DeepRectangling) | — |
| 2 | [**Automatic Color Image Stitching Using Quaternion Rank-1 Alignment**](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Automatic_Color_Image_Stitching_Using_Quaternion_Rank-1_Alignment_CVPR_2022_paper.html) | — | — |
| 3 | [**Geometric Structure Preserving Warp for Natural Image Stitching**](https://openaccess.thecvf.com/content/CVPR2022/html/Du_Geometric_Structure_Preserving_Warp_for_Natural_Image_Stitching_CVPR_2022_paper.html) | — | — |

Image Compression – image compression
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Neural Data-Dependent Transform for Learned Image Compression**](https://arxiv.org/abs/2203.04963v1) | — | — |
| 2 | [**The Devil Is in the Details: Window-based Attention for Image Compression**](https://arxiv.org/abs/2203.08450) | [💻 Code](https://github.com/Googolxx/STF) | — |
| 3 | [**ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding**](https://arxiv.org/abs/2203.10886) | — | — |
| 4 | [**Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression**](https://arxiv.org/abs/2203.10897) | [💻 Code](https://github.com/xiaosu-zhu/McQuic) | — |
| 5 | [**DPICT: Deep Progressive Image Compression Using Trit-Planes**](https://arxiv.org/abs/2112.06334) | [💻 Code](https://github.com/jaehanlee-mcl/DPICT) | — |
| 6 | [**Joint Global and Local Hierarchical Priors for Learned Image Compression**](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Joint_Global_and_Local_Hierarchical_Priors_for_Learned_Image_Compression_CVPR_2022_paper.html) | — | — |
| 7 | [**LC-FDNet: Learned Lossless Image Compression With Frequency Decomposition Network**](https://openaccess.thecvf.com/content/CVPR2022/html/Rhee_LC-FDNet_Learned_Lossless_Image_Compression_With_Frequency_Decomposition_Network_CVPR_2022_paper.html) | — | — |
| 8 | [**Practical Learned Lossless JPEG Recompression with Multi-Level Cross-Channel Entropy Model in the DCT Domain**](https://arxiv.org/abs/2203.16357) | — | `Compress JPEG` |
| 9 | [**SASIC: Stereo Image Compression With Latent Shifts and Stereo Attention**](https://openaccess.thecvf.com/content/CVPR2022/html/Wodlinger_SASIC_Stereo_Image_Compression_With_Latent_Shifts_and_Stereo_Attention_CVPR_2022_paper.html) | — | `Stereo Image Compression` |
| 10 | [**Deep Stereo Image Compression via Bi-Directional Coding**](https://openaccess.thecvf.com/content/CVPR2022/html/Lei_Deep_Stereo_Image_Compression_via_Bi-Directional_Coding_CVPR_2022_paper.html) | — | `Stereo Image Compression` |
| 11 | [**Learning Based Multi-Modality Image and Video Compression**](https://openaccess.thecvf.com/content/CVPR2022/html/Lu_Learning_Based_Multi-Modality_Image_and_Video_Compression_CVPR_2022_paper.html) | — | — |
| 12 | [**PO-ELIC: Perception-Oriented Efficient Learned Image Coding**](https://arxiv.org/abs/2205.14501) | — | `\[Workshop\]` |

### Video Compression

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Coarse-to-fine Deep Video Coding with Hyperprior-guided Mode Prediction**](https://arxiv.org/abs/2206.07460) | — | — |
| 2 | [**LSVC: A Learning-Based Stereo Video Compression Framework**](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_LSVC_A_Learning-Based_Stereo_Video_Compression_Framework_CVPR_2022_paper.html) | — | `Stereo Video Compression` |
| 3 | [**Enhancing VVC with Deep Learning based Multi-Frame Post-Processing**](https://arxiv.org/abs/2205.09458) | — | `\[Workshop\]` |

Image Quality Assessment – ​​image quality assessment
-----------------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Personalized Image Aesthetics Assessment with Rich Attributes**](https://arxiv.org/abs/2203.16754) | — | `Aesthetics Assessment` |
| 2 | [**Incorporating Semi-Supervised and Positive-Unlabeled Learning for Boosting Full Reference Image Quality Assessment**](https://arxiv.org/abs/2204.08763) | [💻 Code](https://github.com/happycaoyue/JSPL) | `FR-IQA` |
| 3 | [**SwinIQA: Learned Swin Distance for Compressed Image Quality Assessment**](https://arxiv.org/abs/2205.04264) | — | `\[Workshop\], compressed IQA` |

Image Decomposition
-------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**PIE-Net: Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition**](https://openaccess.thecvf.com/content/CVPR2022/html/Das_PIE-Net_Photometric_Invariant_Edge_Guided_Network_for_Intrinsic_Image_Decomposition_CVPR_2022_paper.html) | [💻 Code](https://github.com/Morpheus3000/PIE-Net) | — |
| 2 | [**Deformable Sprites for Unsupervised Video Decomposition**](https://arxiv.org/abs/2204.07151) | [💻 Code](https://github.com/vye16/deformable-sprites) | — |

Style Transfer – style transfer
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**CLIPstyler: Image Style Transfer with a Single Text Condition**](https://arxiv.org/abs/2112.00374) | [💻 Code](https://github.com/cyclomon/CLIPstyler) | `CLIP` |
| 2 | [**Style-ERD: Responsive and Coherent Online Motion Style Transfer**](https://arxiv.org/abs/2203.02574) | [💻 Code](https://github.com/tianxintao/Online-Motion-Style-Transfer) | — |
| 3 | [**Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization**](https://arxiv.org/abs/2203.07740) | [💻 Code](https://github.com/YBZh/EFDM) | — |
| 4 | [**Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer**](https://arxiv.org/abs/2203.13248) | [💻 Code](https://github.com/williamyang1991/DualStyleGAN) | — |
| 5 | [**Industrial Style Transfer with Large-scale Geometric Warping and Content Preservation**](https://arxiv.org/abs/2203.12835) | [💻 Code](https://github.com/jcyang98/InST) | — |
| 6 | [**StyTr2: Image Style Transfer With Transformers**](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_StyTr2_Image_Style_Transfer_With_Transformers_CVPR_2022_paper.html) | — | — |
| 7 | [**PCA-Based Knowledge Distillation Towards Lightweight and Content-Style Balanced Photorealistic Style Transfer Models**](https://arxiv.org/abs/2203.13452) | [💻 Code](https://github.com/chiutaiyin/PCA-Knowledge-Distillation) | — |

Image Editing – image editing
-----------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**High-Fidelity GAN Inversion for Image Attribute Editing**](https://arxiv.org/abs/2109.06590) | [💻 Code](https://github.com/Tengfei-Wang/HFGI) | — |
| 2 | [**Style Transformer for Image Inversion and Editing**](https://arxiv.org/abs/2203.07932) | [💻 Code](https://github.com/sapphire497/style-transformer) | — |
| 3 | [**HairCLIP: Design Your Hair by Text and Reference Image**](https://arxiv.org/abs/2112.05142) | [💻 Code](https://github.com/wty-ustc/HairCLIP) | `CLIP` |
| 4 | [**HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing**](https://arxiv.org/abs/2111.15666) | [💻 Code](https://github.com/yuval-alaluf/hyperstyle) | — |
| 5 | [**Blended Diffusion for Text-driven Editing of Natural Images**](https://arxiv.org/abs/2111.14818) | [💻 Code](https://github.com/omriav/blended-diffusion) | `CLIP, Diffusion Model` |
| 6 | [**FlexIT: Towards Flexible Semantic Image Translation**](https://arxiv.org/abs/2203.04705) | — | — |
| 7 | [**SemanticStyleGAN: Learning Compositonal Generative Priors for Controllable Image Synthesis and Editing**](https://arxiv.org/abs/2112.02236) | — | — |
| 8 | [**SketchEdit: Mask-Free Local Image Manipulation with Partial Sketches**](https://arxiv.org/abs/2111.15078) | [💻 Code](https://github.com/zengxianyu/sketchedit) | — |
| 9 | [**TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing**](https://arxiv.org/abs/2203.17266) | [💻 Code](https://github.com/BillyXYB/TransEditor) | — |
| 10 | [**HyperInverter: Improving StyleGAN Inversion via Hypernetwork**](https://arxiv.org/abs/2112.00719) | [💻 Code](https://github.com/VinAIResearch/HyperInverter) | — |
| 11 | [**Spatially-Adaptive Multilayer Selection for GAN Inversion and Editing**](https://arxiv.org/abs/2206.08357) | [💻 Code](https://github.com/adobe-research/sam_inversion) | — |
| 12 | [**Brain-Supervised Image Editing**](https://openaccess.thecvf.com/content/CVPR2022/html/Davis_Brain-Supervised_Image_Editing_CVPR_2022_paper.html) | — | — |
| 13 | [**SpaceEdit: Learning a Unified Editing Space for Open-Domain Image Color Editing**](https://openaccess.thecvf.com/content/CVPR2022/html/Shi_SpaceEdit_Learning_a_Unified_Editing_Space_for_Open-Domain_Image_Color_CVPR_2022_paper.html) | — | — |
| 14 | [**M3L: Language-based Video Editing via Multi-Modal Multi-Level Transformers**](https://arxiv.org/abs/2104.01122) | — | — |

Image Generation/Synthesis / Image-to-Image Translation – Image Generation/Synthesis/Translation
------------------------------------------------------------------------------------------------

### Text-to-Image / Text Guided / Multi-Modal

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Text to Image Generation with Semantic-Spatial Aware GAN**](https://arxiv.org/abs/2104.00567) | [💻 Code](https://github.com/wtliao/text2image) | — |
| 2 | [**LAFITE: Towards Language-Free Training for Text-to-Image Generation**](https://arxiv.org/abs/2111.13792) | [💻 Code](https://github.com/drboog/Lafite) | — |
| 3 | [**DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis**](https://arxiv.org/abs/2008.05865) | [💻 Code](https://github.com/tobran/DF-GAN) | — |
| 4 | [**StyleT2I: Toward Compositional and High-Fidelity Text-to-Image Synthesis**](https://arxiv.org/abs/2203.15799) | [💻 Code](https://github.com/zhihengli-UR/StyleT2I) | — |
| 5 | [**DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation**](https://arxiv.org/abs/2110.02711) | [💻 Code](https://github.com/gwang-kim/DiffusionCLIP) | — |
| 6 | [**Predict, Prevent, and Evaluate: Disentangled Text-Driven Image Manipulation Empowered by Pre-Trained Vision-Language Model**](https://arxiv.org/abs/2111.13333) | [💻 Code](https://github.com/zipengxuc/PPE-Pytorch) | — |
| 7 | [**Sound-Guided Semantic Image Manipulation**](https://arxiv.org/abs/2112.00007) | [💻 Code](https://github.com/kuai-lab/sound-guided-semantic-image-manipulation) | — |
| 8 | [**ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation**](https://arxiv.org/abs/2204.04428) | — | — |
| 9 | [**Text-to-Image Synthesis Based on Object-Guided Joint-Decoding Transformer**](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Text-to-Image_Synthesis_Based_on_Object-Guided_Joint-Decoding_Transformer_CVPR_2022_paper.html) | — | — |
| 10 | [**Vector Quantized Diffusion Model for Text-to-Image Synthesis**](https://arxiv.org/abs/2111.14822) | — | — |
| 11 | [**AnyFace: Free-style Text-to-Face Synthesis and Manipulation**](https://arxiv.org/abs/2203.15334) | — | — |

### Image-to-Image / Image Guided

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation**](https://arxiv.org/abs/2203.12707) | [💻 Code](https://github.com/batmanlab/MSPC) | — |
| 2 | [**A Style-aware Discriminator for Controllable Image Translation**](https://arxiv.org/abs/2203.15375) | [💻 Code](https://github.com/kunheek/style-aware-discriminator) | — |
| 3 | [**QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation**](https://arxiv.org/abs/2203.08483) | [💻 Code](https://github.com/sapphire497/query-selected-attention) | — |
| 4 | [**InstaFormer: Instance-Aware Image-to-Image Translation with Transformer**](https://arxiv.org/abs/2203.16248) | — | — |
| 5 | [**Marginal Contrastive Correspondence for Guided Image Generation**](https://arxiv.org/abs/2204.00442) | [💻 Code](https://github.com/fnzhan/UNITE) | — |
| 6 | [**Unsupervised Image-to-Image Translation with Generative Prior**](https://arxiv.org/abs/2204.03641) | [💻 Code](https://github.com/williamyang1991/GP-UNIT) | — |
| 7 | [**Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks**](https://arxiv.org/abs/2203.01532) | [💻 Code](https://github.com/jcy132/Hneg_SRC) | — |
| 8 | [**Neural Texture Extraction and Distribution for Controllable Person Image Synthesis**](https://arxiv.org/abs/2204.06160) | [💻 Code](https://github.com/RenYurui/Neural-Texture-Extraction-Distribution) | — |
| 9 | [**Unpaired Cartoon Image Synthesis via Gated Cycle Mapping**](https://openaccess.thecvf.com/content/CVPR2022/html/Men_Unpaired_Cartoon_Image_Synthesis_via_Gated_Cycle_Mapping_CVPR_2022_paper.html) | — | — |
| 10 | [**Day-to-Night Image Synthesis for Training Nighttime Neural ISPs**](https://openaccess.thecvf.com/content/CVPR2022/html/Punnappurath_Day-to-Night_Image_Synthesis_for_Training_Nighttime_Neural_ISPs_CVPR_2022_paper.html) | [💻 Code](https://github.com/SamsungLabs/day-to-night) | — |
| 11 | [**Alleviating Semantics Distortion in Unsupervised Low-Level Image-to-Image Translation via Structure Consistency Constraint**](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Alleviating_Semantics_Distortion_in_Unsupervised_Low-Level_Image-to-Image_Translation_via_Structure_CVPR_2022_paper.html) | — | — |
| 12 | [**Wavelet Knowledge Distillation: Towards Efficient Image-to-Image Translation**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Wavelet_Knowledge_Distillation_Towards_Efficient_Image-to-Image_Translation_CVPR_2022_paper.html) | — | — |
| 13 | [**Self-Supervised Dense Consistency Regularization for Image-to-Image Translation**](https://openaccess.thecvf.com/content/CVPR2022/html/Ko_Self-Supervised_Dense_Consistency_Regularization_for_Image-to-Image_Translation_CVPR_2022_paper.html) | — | — |
| 14 | [**Drop the GAN: In Defense of Patches Nearest Neighbors as Single Image Generative Model**](https://arxiv.org/abs/2103.15545) | — | `image manipulation` |
| 15 | [**HairMapper: Removing Hair From Portraits Using GANs**](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_HairMapper_Removing_Hair_From_Portraits_Using_GANs_CVPR_2022_paper.html) | — | — |

### Others for image generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Attribute Group Editing for Reliable Few-shot Image Generation**](https://arxiv.org/abs/2203.08422) | [💻 Code](https://github.com/UniBester/AGE) | — |
| 2 | [**Modulated Contrast for Versatile Image Synthesis**](https://arxiv.org/abs/2203.09333) | [💻 Code](https://github.com/fnzhan/MoNCE) | — |
| 3 | [**Interactive Image Synthesis with Panoptic Layout Generation**](https://arxiv.org/abs/2203.02104) | — | — |
| 4 | [**Autoregressive Image Generation using Residual Quantization**](https://arxiv.org/abs/2203.01941) | [💻 Code](https://github.com/lucidrains/RQ-Transformer) | — |
| 5 | [**Dynamic Dual-Output Diffusion Models**](https://arxiv.org/abs/2203.04304) | — | — |
| 6 | [**Exploring Dual-task Correlation for Pose Guided Person Image Generation**](https://arxiv.org/abs/2203.02910) | [💻 Code](https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network) | — |
| 7 | [**StyleSwin: Transformer-based GAN for High-resolution Image Generation**](https://arxiv.org/abs/2112.10762) | [💻 Code](https://github.com/microsoft/StyleSwin) | — |
| 8 | [**Semantic-shape Adaptive Feature Modulation for Semantic Image Synthesis**](https://arxiv.org/abs/2203.16898) | [💻 Code](https://github.com/cszy98/SAFM) | — |
| 9 | [**Arbitrary-Scale Image Synthesis**](https://arxiv.org/abs/2204.02273) | [💻 Code](https://github.com/vglsd/ScaleParty) | — |
| 10 | [**InsetGAN for Full-Body Image Generation**](https://arxiv.org/abs/2203.07293) | — | — |
| 11 | [**HairMapper: Removing Hair from Portraits Using GANs**](http://www.cad.zju.edu.cn/home/jin/cvpr2022/HairMapper.pdf) | [💻 Code](https://github.com/oneThousand1000/non-hair-FFHQ) | — |
| 12 | [**OSSGAN: Open-Set Semi-Supervised Image Generation**](https://arxiv.org/abs/2204.14249) | [💻 Code](https://github.com/raven38/OSSGAN) | — |
| 13 | [**Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis**](https://arxiv.org/abs/2204.02854) | [💻 Code](https://github.com/Shi-Yupeng/RESAIL-For-SIS) | — |
| 14 | [**A Closer Look at Few-shot Image Generation**](https://arxiv.org/abs/2205.03805) | — | `Few-shot` |
| 15 | [**Ensembling Off-the-shelf Models for GAN Training**](https://arxiv.org/abs/2112.09130) | [💻 Code](https://github.com/nupurkmr9/vision-aided-gan) | — |
| 16 | [**Few-Shot Font Generation by Learning Fine-Grained Local Styles**](https://arxiv.org/abs/2205.09965) | — | `Few-shot` |
| 17 | [**Modeling Image Composition for Complex Scene Generation**](https://arxiv.org/abs/2206.00923) | [💻 Code](https://github.com/JohnDreamer/TwFA) | — |
| 18 | [**Global Context With Discrete Diffusion in Vector Quantized Modeling for Image Generation**](https://arxiv.org/abs/2112.01799) | — | — |
| 19 | [**Self-supervised Correlation Mining Network for Person Image Generation**](https://arxiv.org/abs/2111.13307) | — | — |
| 20 | [**Learning To Memorize Feature Hallucination for One-Shot Image Generation**](https://openaccess.thecvf.com/content/CVPR2022/html/Xie_Learning_To_Memorize_Feature_Hallucination_for_One-Shot_Image_Generation_CVPR_2022_paper.html) | — | — |
| 21 | [**Local Attention Pyramid for Scene Image Generation**](https://openaccess.thecvf.com/content/CVPR2022/html/Shim_Local_Attention_Pyramid_for_Scene_Image_Generation_CVPR_2022_paper.html) | — | — |
| 22 | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752) | [💻 Code](https://github.com/CompVis/latent-diffusion) | — |
| 23 | [**Cluster-guided Image Synthesis with Unconditional Models**](https://arxiv.org/abs/2112.12911) | — | — |
| 24 | [**SphericGAN: Semi-Supervised Hyper-Spherical Generative Adversarial Networks for Fine-Grained Image Synthesis**](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_SphericGAN_Semi-Supervised_Hyper-Spherical_Generative_Adversarial_Networks_for_Fine-Grained_Image_Synthesis_CVPR_2022_paper.html) | — | — |
| 25 | [**DPGEN: Differentially Private Generative Energy-Guided Network for Natural Image Synthesis**](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_DPGEN_Differentially_Private_Generative_Energy-Guided_Network_for_Natural_Image_Synthesis_CVPR_2022_paper.html) | — | — |
| 26 | [**DO-GAN: A Double Oracle Framework for Generative Adversarial Networks**](https://openaccess.thecvf.com/content/CVPR2022/html/Aung_DO-GAN_A_Double_Oracle_Framework_for_Generative_Adversarial_Networks_CVPR_2022_paper.html) | — | — |
| 27 | [**Improving GAN Equilibrium by Raising Spatial Awareness**](https://arxiv.org/abs/2112.00718) | [💻 Code](https://github.com/genforce/eqgan-sa) | — |

\*\*Polymorphic-GAN: Generating Aligned Samples Across Multiple Domains With Learned Morph Maps\*\*

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Polymorphic-GAN_Generating_Aligned_Samples_Across_Multiple_Domains_With_Learned_Morph_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Manifold Learning Benefits GANs**](https://arxiv.org/abs/2112.12618) | — | — |
| 2 | [**Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data**](https://arxiv.org/abs/2204.04950) | [💻 Code](https://github.com/FriedRonaldo/Primitives-PS) | — |
| 3 | [**On Conditioning the Input Noise for Controlled Image Generation with Diffusion Models**](https://arxiv.org/abs/2205.03859) | — | `\[Workshop\]` |
| 4 | [**Generate and Edit Your Own Character in a Canonical View**](https://arxiv.org/abs/2205.02974) | — | `\[Workshop\]` |
| 5 | [**StyLandGAN: A StyleGAN based Landscape Image Synthesis using Depth-map**](https://arxiv.org/abs/2205.06611) | — | `\[Workshop\]` |
| 6 | [**Overparameterization Improves StyleGAN Inversion**](https://arxiv.org/abs/2205.06304) | — | `\[Workshop\]` |

### Video Generation/Synthesis

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning**](https://arxiv.org/abs/2203.02573) | [💻 Code](https://github.com/snap-research/MMVID) | — |
| 2 | [**Playable Environments: Video Manipulation in Space and Time**](https://arxiv.org/abs/2203.01914) | [💻 Code](https://github.com/willi-menapace/PlayableEnvironments) | — |
| 3 | [**StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2**](https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf) | [💻 Code](https://github.com/universome/stylegan-v) | — |
| 4 | [**Thin-Plate Spline Motion Model for Image Animation**](https://arxiv.org/abs/2203.14367) | [💻 Code](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model) | — |
| 5 | [**Make It Move: Controllable Image-to-Video Generation with Text Descriptions**](https://arxiv.org/abs/2112.02815) | [💻 Code](https://github.com/Youncy-Hu/MAGE) | — |
| 6 | [**Diverse Video Generation from a Single Video**](https://arxiv.org/abs/2205.05725) | — | `\[Workshop\]` |

Others
------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**GAN-Supervised Dense Visual Alignment**](https://arxiv.org/abs/2112.05143) | [💻 Code](https://github.com/wpeebles/gangealing) | — |
| 2 | [**ClothFormer: Taming Video Virtual Try-on in All Module**](https://arxiv.org/abs/2204.12151) | — | `Video Virtual Try-on` |
| 3 | [**Iterative Deep Homography Estimation**](https://arxiv.org/abs/2203.15982) | [💻 Code](https://github.com/imdumpl78/IHN) | — |
| 4 | [**Style-Structure Disentangled Features and Normalizing Flows for Diverse Icon Colorization**](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Style-Structure_Disentangled_Features_and_Normalizing_Flows_for_Diverse_Icon_Colorization_CVPR_2022_paper.pdf) | [💻 Code](https://github.com/djosix/IconFlow) | — |
| 5 | [**Unsupervised Homography Estimation with Coplanarity-Aware GAN**](https://arxiv.org/abs/2205.03821) | [💻 Code](https://github.com/megvii-research/HomoGAN) | — |
| 6 | [**Diverse Image Outpainting via GAN Inversion**](https://arxiv.org/abs/2104.00675) | [💻 Code](https://github.com/yccyenchicheng/InOut) | — |
| 7 | [**On Aliased Resizing and Surprising Subtleties in GAN Evaluation**](https://arxiv.org/abs/2104.11222) | [💻 Code](https://github.com/GaParmar/clean-fid) | — |
| 8 | [**Patch-wise Contrastive Style Learning for Instagram Filter Removal**](https://arxiv.org/abs/2204.07486) | [💻 Code](https://github.com/birdortyedi/cifr-pytorch) | `\[Workshop\]` |

NTIRE2022
---------

New Trends in Image Restoration and Enhancement workshop and challenges on image and video processing.

### Spectral Reconstruction from RGB

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction**](https://arxiv.org/abs/2204.07908) | [💻 Code](https://github.com/caiyuanhao1998/MST-plus-plus) | `1st place` |

### Perceptual Image Quality Assessment: Track 1 Full-Reference / Track 2 No-Reference

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment**](https://arxiv.org/abs/2204.08958) | [💻 Code](https://github.com/IIGROUP/MANIQA) | `1st place for track2` |
| 2 | [**Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network**](https://arxiv.org/abs/2204.10485) | [💻 Code](https://github.com/IIGROUP/AHIQ) | `1st place for track1` |
| 3 | [**MSTRIQ: No Reference Image Quality Assessment Based on Swin Transformer with Multi-Stage Fusion**](https://arxiv.org/abs/2205.10101) | — | `2nd place in track2` |
| 4 | [**Conformer and Blind Noisy Students for Improved Image Quality Assessment**](https://arxiv.org/abs/2204.12819) | — | — |

### Inpainting: Track 1 Unsupervised / Track 2 Semantic

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**GLaMa: Joint Spatial and Frequency Loss for General Image Inpainting**](https://arxiv.org/abs/2205.07162) | — | `ranked first in terms of PSNR, LPIPS and SSIM in the track1` |

### Efficient Super-Resolution

*   **Report** :  [https://arxiv.org/abs/2205.05675](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.05675 "https://arxiv.org/abs/2205.05675")

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ShuffleMixer: An Efficient ConvNet for Image Super-Resolution**](https://arxiv.org/abs/2205.15175) | [💻 Code](https://github.com/sunny2109/MobileSR-NTIRE2022) | `Winner of the model complexity track` |
| 2 | [**Edge-enhanced Feature Distillation Network for Efficient Super-Resolution**](https://arxiv.org/abs/2204.08759) | [💻 Code](https://github.com/icandle/EFDN) | — |
| 3 | [**Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution**](https://arxiv.org/abs/2204.08759) | [💻 Code](https://github.com/NJU-Jet/FMEN) | `Lowest memory consumption and second shortest runtime` |
| 4 | [**Blueprint Separable Residual Network for Efficient Image Super-Resolution**](https://arxiv.org/abs/2205.05996) | [💻 Code](https://github.com/xiaom233/BSRN) | `1st place in model complexity track` |

### Night Photography Rendering

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Rendering Nighttime Image Via Cascaded Color and Brightness Compensation**](https://arxiv.org/abs/2204.08970) | [💻 Code](https://github.com/NJUVISION/CBUnet) | `2nd place` |

### Super-Resolution and Quality Enhancement of Compressed Video: Track1 (Quality enhancement) / Track2 (Quality enhancement and x2 SR) / Track3 (Quality enhancement and x4 SR)

*   **Report** :  [https://arxiv.org/abs/2204.09314](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.09314 "https://arxiv.org/abs/2204.09314")
*   **Homepage** :  [GitHub – RenYang-home/NTIRE22\_VEnh\_SR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/RenYang-home/NTIRE22_VEnh_SR "GitHub - RenYang-home/NTIRE22_VEnh_SR")

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Progressive Training of A Two-Stage Framework for Video Restoration**](https://arxiv.org/abs/2204.09924) | [💻 Code](https://github.com/ryanxingql/winner-ntire22-vqe) | `1st place in track1 and track2, 2nd place in track3` |

### High Dynamic Range (HDR): Track 1 Low-complexity (fidelity constraint) / Track 2 Fidelity (low-complexity constraint)

*   **Report** :  [https://arxiv.org/abs/2205.12633](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.12633 "https://arxiv.org/abs/2205.12633")

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Efficient Progressive High Dynamic Range Image Restoration via Attention and Alignment Network**](https://arxiv.org/abs/2204.09213) | — | `2nd palce of both two tracks` |

### Stereo Super-Resolution

*   **Report** :  [https://arxiv.org/abs/2204.09197](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.09197 "https://arxiv.org/abs/2204.09197")

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | **Parallel Interactive Transformer** | [💻 Code](https://github.com/chaineypung/CVPR-NTIRE2022-Parallel-Interactive-Transformer-PAIT) | `7st place` |

### Burst Super-Resolution: Track 2 Real

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | **BSRT: Improving Burst Super-Resolution with Swin Transformer and Flow-Guided Deformable Alignment** | [💻 Code](https://github.com/Algolzw/BSRT) | `1st place` |

ECCV2022-Low-Level-Vision
=========================

Image Restoration – Image Restoration
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Simple Baselines for Image Restoration**](https://arxiv.org/abs/2204.04676) | [💻 Code](https://github.com/megvii-research/NAFNet) | — |
| 2 | [**D2HNet: Joint Denoising and Deblurring with Hierarchical Network for Robust Night Image Restoration**](https://arxiv.org/abs/2207.03294) | [💻 Code](https://github.com/zhaoyuzhi/D2HNet) | — |
| 3 | [**Seeing Far in the Dark with Patterned Flash**](https://arxiv.org/abs/2207.12570) | [💻 Code](https://github.com/zhsun0357/Seeing-Far-in-the-Dark-with-Patterned-Flash) | — |
| 4 | [**BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks**](https://arxiv.org/abs/2207.06873) | [💻 Code](https://github.com/ExplainableML/BayesCap) | — |
| 5 | [**Improving Image Restoration by Revisiting Global Information Aggregation**](https://arxiv.org/abs/2112.04491) | [💻 Code](https://github.com/megvii-research/TLC) | — |
| 6 | [**Fast Two-step Blind Optical Aberration Correction**](https://arxiv.org/abs/2208.00950) | [💻 Code](https://github.com/teboli/fast_two_stage_psf_correction) | `Optical Aberration Correction` |
| 7 | [**VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder**](https://arxiv.org/abs/2205.06803) | [💻 Code](https://github.com/TencentARC/VQFR) | `Blind Face Restoration` |
| 8 | [**RAWtoBit: A Fully End-to-end Camera ISP Network**](https://arxiv.org/abs/2208.07639) | — | `ISP and Image Compression` |
| 9 | [**Transform your Smartphone into a DSLR Camera: Learning the ISP in the Wild**](https://arxiv.org/abs/2203.10636) | [💻 Code](https://github.com/4rdhendu/TransformPhone2DSLR) | `ISP` |
| 10 | [**Single Frame Atmospheric Turbulence Mitigation: A Benchmark Study and A New Physics-Inspired Transformer Model**](https://arxiv.org/abs/2207.10040) | [💻 Code](https://github.com/VITA-Group/TurbNet) | `Atmospheric Turbulence Mitigation, Transformer` |
| 11 | [**Modeling Mask Uncertainty in Hyperspectral Image Reconstruction**](https://arxiv.org/abs/2112.15362) | [💻 Code](https://github.com/Jiamian-Wang/mask_uncertainty_spectral_SCI) | `Hyperspectral Image Reconstruction` |
| 12 | [**TAPE: Task-Agnostic Prior Embedding for Image Restoration**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3292_ECCV_2022_paper.php) | — | — |
| 13 | [**DRCNet: Dynamic Image Restoration Contrastive Network**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6389_ECCV_2022_paper.php) | — | — |
| 14 | [**ART-SS: An Adaptive Rejection Technique for Semi-Supervised Restoration for Adverse Weather-Affected Images**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4237_ECCV_2022_paper.php) | [💻 Code](https://github.com/rajeevyasarla/ART-SS) | `Adverse Weather` |
| 15 | [**Spectrum-Aware and Transferable Architecture Search for Hyperspectral Image Restoration**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4367_ECCV_2022_paper.php) | — | `Hyperspectral Image Restoration` |
| 16 | [**Seeing through a Black Box: Toward High-Quality Terahertz Imaging via Subspace-and-Attention Guided Restoration**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6259_ECCV_2022_paper.php) | — | `Terahertz Imaging` |
| 17 | [**JPEG Artifacts Removal via Contrastive Representation Learning**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/171_ECCV_2022_paper.php) | — | `JPEG Artifacts Removal` |
| 18 | [**Zero-Shot Learning for Reflection Removal of Single 360-Degree Image**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6418_ECCV_2022_paper.php) | — | `Reflection Removal` |
| 19 | [**Overexposure Mask Fusion: Generalizable Reverse ISP Multi-Step Refinement**](https://arxiv.org/abs/2210.11511) | [💻 Code](https://github.com/SenseBrainTech/overexposure-mask-reverse-ISP) | — |

### Video Restoration

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Video Restoration Framework and Its Meta-Adaptations to Data-Poor Conditions**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7533_ECCV_2022_paper.php) | — | — |

Super Resolution – super resolution
-----------------------------------

### Image Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ARM: Any-Time Super-Resolution Method**](https://arxiv.org/abs/2203.10812) | [💻 Code](https://github.com/chenbong/ARM-Net) | — |
| 2 | [**Dynamic Dual Trainable Bounds for Ultra-low Precision Super-Resolution Networks**](https://arxiv.org/abs/2203.03844) | [💻 Code](https://github.com/zysxmu/DDTB) | — |
| 3 | [**CADyQ : Contents-Aware Dynamic Quantization for Image Super Resolution**](https://arxiv.org/abs/2207.10345) | [💻 Code](https://github.com/Cheeun/CADyQ) | — |
| 4 | [**Image Super-Resolution with Deep Dictionary**](https://arxiv.org/abs/2207.09228) | [💻 Code](https://github.com/shuntama/srdd) | — |
| 5 | [**Perception-Distortion Balanced ADMM Optimization for Single-Image Super-Resolution**](https://arxiv.org/abs/2208.03324) | [💻 Code](https://github.com/Yuehan717/PDASR) | — |
| 6 | [**Adaptive Patch Exiting for Scalable Single Image Super-Resolution**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2021_ECCV_2022_paper.php) | [💻 Code](https://github.com/littlepure2333/APE) | — |
| 7 | [**Learning Series-Parallel Lookup Tables for Efficient Image Super-Resolution**](https://arxiv.org/abs/2207.12987) | [💻 Code](https://github.com/zhjy2016/SPLUT) | `Efficient` |
| 8 | [**MuLUT: Cooperating Mulitple Look-Up Tables for Efficient Image Super-Resolution**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780234.pdf) | [💻 Code](https://github.com/ddlee-cn/MuLUT) | `Efficient` |
| 9 | [**Efficient Long-Range Attention Network for Image Super-resolution**](https://arxiv.org/abs/2203.06697) | [💻 Code](https://github.com/xindongzhang/ELAN) | — |
| 10 | [**Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution**](https://arxiv.org/abs/2207.12577) | [💻 Code](https://github.com/wuyushuwys/compiler-aware-nas-sr) | — |
| 11 | [**Restore Globally, Refine Locally: A Mask-Guided Scheme to Accelerate Super-Resolution Networks**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4417_ECCV_2022_paper.php) | [💻 Code](https://github.com/huxiaotaostasy/MGA-scheme) | — |
| 12 | [**Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution**](https://arxiv.org/abs/2207.09156) | [💻 Code](https://github.com/palmdong/MMSR) | `Self-Supervised` |
| 13 | [**Self-Supervised Learning for Real-World Super-Resolution from Dual Zoomed Observations**](https://arxiv.org/abs/2203.01325) | [💻 Code](https://github.com/cszhilu1998/SelfDZSR) | `Self-Supervised, Reference-based` |
| 14 | [**Efficient and Degradation-Adaptive Network for Real-World Image Super-Resolution**](http://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022_DASR.pdf) | [💻 Code](https://github.com/csjliang/DASR) | `Real World` |
| 15 | [**D2C-SR: A Divergence to Convergence Approach for Real-World Image Super-Resolution**](https://arxiv.org/abs/2103.14373) | [💻 Code](https://github.com/megvii-research/D2C-SR) | `Real World` |
| 16 | [**MM-RealSR: Metric Learning based Interactive Modulation for Real-World Super-Resolution**](https://arxiv.org/abs/2205.05065) | [💻 Code](https://github.com/TencentARC/MM-RealSR) | `Real World` |
| 17 | [**KXNet: A Model-Driven Deep Neural Network for Blind Super-Resolution**](https://arxiv.org/abs/2209.10305) | [💻 Code](https://github.com/jiahong-fu/KXNet) | `Blind` |
| 18 | [**From Face to Natural Image: Learning Real Degradation for Blind Image Super-Resolution**](https://arxiv.org/abs/2210.00752) | [💻 Code](https://github.com/csxmli2016/ReDegNet) | `Blind` |
| 19 | [**Unfolded Deep Kernel Estimation for Blind Image Super-Resolution**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3484_ECCV_2022_paper.php) | [💻 Code](https://github.com/natezhenghy/UDKE) | `Blind` |
| 20 | [**Uncertainty Learning in Kernel Estimation for Multi-stage Blind Image Super-Resolution**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1649_ECCV_2022_paper.php) | — | `Blind` |
| 21 | [**Super-Resolution by Predicting Offsets: An Ultra-Efficient Super-Resolution Network for Rasterized Images**](https://arxiv.org/abs/2210.04198) | [💻 Code](https://github.com/HaomingCai/SRPO) | `Rasterized Images` |
| 22 | [**Reference-based Image Super-Resolution with Deformable Attention Transformer**](https://arxiv.org/abs/2207.11938) | [💻 Code](https://github.com/caojiezhang/DATSR) | `Reference-based, Transformer` |
| 23 | [**RRSR: Reciprocal Reference-Based Image Super-Resolution with Progressive Feature Alignment and Selection**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7808_ECCV_2022_paper.php) | — | `Reference-based` |
| 24 | [**Boosting Event Stream Super-Resolution with a Recurrent Neural Network**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/248_ECCV_2022_paper.php) | — | `Event` |
| 25 | [**HST: Hierarchical Swin Transformer for Compressed Image Super-resolution**](https://arxiv.org/abs/2208.09885) | — | `\[Workshop-AIM2022\]` |
| 26 | [**Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration**](https://arxiv.org/abs/2209.11345) | [💻 Code](https://github.com/mv-lab/swin2sr) | `\[Workshop-AIM2022\]` |
| 27 | [**Fast Nearest Convolution for Real-Time Efficient Image Super-Resolution**](https://arxiv.org/abs/2208.11609) | [💻 Code](https://github.com/Algolzw/NCNet) | `\[Workshop-AIM2022\]` |

### Video Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning Spatiotemporal Frequency-Transformer for Compressed Video Super-Resolution**](https://arxiv.org/abs/2208.03012) | [💻 Code](https://github.com/researchmm/FTVSR) | `Compressed Video SR` |
| 2 | [**A Codec Information Assisted Framework for Efficient Compressed Video Super-Resolution**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6420_ECCV_2022_paper.php) | — | `Compressed Video SR` |
| 3 | [**Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset**](https://arxiv.org/abs/2209.12475) | [💻 Code](https://github.com/zmzhang1998/Real-RawVSR) | — |

Denoising – denoising
---------------------

### Image Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Deep Semantic Statistics Matching (D2SM) Denoising Network**](https://arxiv.org/abs/2207.09302) | [💻 Code](https://github.com/MKFMIKU/d2sm) | — |
| 2 | [**Fast and High Quality Image Denoising via Malleable Convolution**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3257_ECCV_2022_paper.php) | — | — |

### Video Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Unidirectional Video Denoising by Mimicking Backward Recurrent Modules with Look-ahead Forward Ones**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4024_ECCV_2022_paper.php) | [💻 Code](https://github.com/nagejacob/FloRNN) | — |
| 2 | [**TempFormer: Temporally Consistent Transformer for Video Denoising**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6092_ECCV_2022_paper.php) | — | `Transformer` |

Deblurring – Deblurring
-----------------------

### Image Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning Degradation Representations for Image Deblurring**](https://arxiv.org/abs/2208.05244) | [💻 Code](https://github.com/dasongli1/Learning_degradation) | — |
| 2 | [**Stripformer: Strip Transformer for Fast Image Deblurring**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4651_ECCV_2022_paper.php) | — | `Transformer` |
| 3 | [**Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance**](https://arxiv.org/abs/2207.10123) | [💻 Code](https://github.com/zzh-tech/Animation-from-Blur) | `recovering detailed motion from a single motion-blurred image` |
| 4 | [**United Defocus Blur Detection and Deblurring via Adversarial Promotion Learning**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3308_ECCV_2022_paper.php) | [💻 Code](https://github.com/wdzhao123/APL) | `Defocus Blur` |
| 5 | [**Realistic Blur Synthesis for Learning Image Deblurring**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6325_ECCV_2022_paper.php) | — | `Blur Synthesis` |
| 6 | [**Event-based Fusion for Motion Deblurring with Cross-modal Attention**](https://arxiv.org/abs/2112.00167) | [💻 Code](https://github.com/AHupuJR/EFNet) | `Event-based` |
| 7 | [**Event-Guided Deblurring of Unknown Exposure Time Videos**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3601_ECCV_2022_paper.php) | — | `Event-based` |

### Video Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Spatio-Temporal Deformable Attention Network for Video Deblurring**](https://arxiv.org/abs/2207.10852) | [💻 Code](https://github.com/huicongzhang/STDAN) | — |
| 2 | [**Efficient Video Deblurring Guided by Motion Magnitude**](https://arxiv.org/abs/2207.13374) | [💻 Code](https://github.com/sollynoay/MMP-RNN) | — |
| 3 | [**ERDN: Equivalent Receptive Field Deformable Network for Video Deblurring**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4085_ECCV_2022_paper.php) | [💻 Code](https://github.com/TencentCloud/ERDN) | — |
| 4 | [**DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting**](https://arxiv.org/abs/2111.09985) | [💻 Code](https://github.com/JihyongOh/DeMFI) | `Joint Deblurring and Frame Interpolation` |
| 5 | [**Towards Real-World Video Deblurring by Exploring Blur Formation Process**](https://arxiv.org/abs/2208.13184) | — | `\[Workshop-AIM2022\]` |

Image Decomposition
-------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Blind Image Decomposition**](https://arxiv.org/abs/2108.11364) | [💻 Code](https://github.com/JunlinHan/BID) | — |

Deraining – deraining
---------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Not Just Streaks: Towards Ground Truth for Single Image Deraining**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1506_ECCV_2022_paper.php) | [💻 Code](https://github.com/UCLA-VMG/GT-RAIN) | — |
| 2 | [**Rethinking Video Rain Streak Removal: A New Synthesis Model and a Deraining Network with Video Rain Prior**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6798_ECCV_2022_paper.php) | [💻 Code](https://github.com/wangshauitj/RDD-Net) | — |

Dehazing – to fog
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Frequency and Spatial Dual Guidance for Image Dehazing**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4734_ECCV_2022_paper.php) | [💻 Code](https://github.com/yuhuUSTC/FSDGN) | — |
| 2 | [**Perceiving and Modeling Density for Image Dehazing**](https://arxiv.org/abs/2111.09733) | [💻 Code](https://github.com/Owen718/ECCV22-Perceiving-and-Modeling-Density-for-Image-Dehazing) | — |
| 3 | [**Boosting Supervised Dehazing Methods via Bi-Level Patch Reweighting**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1346_ECCV_2022_paper.php) | — | — |
| 4 | [**Unpaired Deep Image Dehazing Using Contrastive Disentanglement Learning**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/255_ECCV_2022_paper.php) | — | — |

Demoireing – Go moiré
---------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing**](https://arxiv.org/abs/2207.09935) | [💻 Code](https://github.com/XinYu-Andy/uhdm-page) | — |

HDR Imaging / Multi-Exposure Image Fusion – HDR image generation / multi-exposure image fusion
----------------------------------------------------------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6250_ECCV_2022_paper.php) | [💻 Code](https://github.com/viengiaan/EDWL) | — |
| 2 | [**Ghost-free High Dynamic Range Imaging with Context-aware Transformer**](https://arxiv.org/abs/2208.05114) | [💻 Code](https://github.com/megvii-research/HDR-Transformer) | — |
| 3 | [**Selective TransHDR: Transformer-Based Selective HDR Imaging Using Ghost Region Mask**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6670_ECCV_2022_paper.php) | — | — |
| 4 | [**HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields**](https://arxiv.org/abs/2208.06787) | [💻 Code](https://github.com/postech-ami/HDR-Plenoxels) | — |
| 5 | [**Towards Real-World HDRTV Reconstruction: A Data Synthesis-Based Approach**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4873_ECCV_2022_paper.php) | — | — |

Image Fusion
------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**FusionVAE: A Deep Hierarchical Variational Autoencoder for RGB Image Fusion**](https://arxiv.org/abs/2209.11277) | — | — |
| 2 | [**Recurrent Correction Network for Fast and Efficient Multi-modality Image Fusion**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3864_ECCV_2022_paper.php) | [💻 Code](https://github.com/MisakiCoca/ReCoNet) | — |
| 3 | [**Neural Image Representations for Multi-Image Fusion and Layer Separation**](https://arxiv.org/abs/2108.01199) | [💻 Code](https://shnnam.github.io/research/nir/) | — |
| 4 | [**Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4260_ECCV_2022_paper.php) | [💻 Code](https://github.com/erfect2020/DecompositionForFusion) | — |

Frame Interpolation – frame insertion
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Real-Time Intermediate Flow Estimation for Video Frame Interpolation**](https://arxiv.org/abs/2011.06294) | [💻 Code](https://github.com/hzwer/ECCV2022-RIFE) | — |
| 2 | [**FILM: Frame Interpolation for Large Motion**](https://arxiv.org/abs/2202.04901) | [💻 Code](https://github.com/google-research/frame-interpolation) | — |
| 3 | [**Video Interpolation by Event-driven Anisotropic Adjustment of Optical Flow**](https://arxiv.org/abs/2208.09127) | — | — |
| 4 | [**Learning Cross-Video Neural Representations for High-Quality Frame Interpolation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2565_ECCV_2022_paper.php) | — | — |
| 5 | [**Deep Bayesian Video Frame Interpolation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1287_ECCV_2022_paper.php) | [💻 Code](https://github.com/Oceanlib/DBVI) | — |
| 6 | [**A Perceptual Quality Metric for Video Frame Interpolation**](https://arxiv.org/abs/2210.01879) | [💻 Code](https://github.com/hqqxyy/VFIPS) | — |
| 7 | [**DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting**](https://arxiv.org/abs/2111.09985) | [💻 Code](https://github.com/JihyongOh/DeMFI) | `Joint Deblurring and Frame Interpolation` |

### Spatial-Temporal Video Super-Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Towards Interpretable Video Super-Resolution via Alternating Optimization**](https://arxiv.org/abs/2207.10765) | [💻 Code](https://github.com/caojiezhang/DAVSR) | — |

Image Enhancement – ​​image enhancement
---------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Local Color Distributions Prior to Image Enhancement**](https://www.cs.cityu.edu.hk/~rynson/papers/eccv22b.pdf) | [💻 Code](https://github.com/hywang99/LCDPNet) | — |
| 2 | [**SepLUT: Separable Image-adaptive Lookup Tables for Real-time Image Enhancement**](https://arxiv.org/abs/2207.08351) | — | — |
| 3 | [**Neural Color Operators for Sequential Image Retouching**](https://arxiv.org/abs/2207.08080) | [💻 Code](https://github.com/amberwangyili/neurop) | — |
| 4 | [**Deep Fourier-Based Exposure Correction Network with Spatial-Frequency Interaction**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4678_ECCV_2022_paper.php) | — | `Exposure Correction` |
| 5 | [**Uncertainty Inspired Underwater Image Enhancement**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3298_ECCV_2022_paper.php) | — | `Underwater Image Enhancement` |
| 6 | [**NEST: Neural Event Stack for Event-Based Image Enhancement**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2730_ECCV_2022_paper.php) | — | `Event-Based` |

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**LEDNet: Joint Low-light Enhancement and Deblurring in the Dark**](https://arxiv.org/abs/2202.03373) | [💻 Code](https://github.com/sczhou/LEDNet) | — |
| 2 | [**Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression**](https://arxiv.org/abs/2207.10564) | [💻 Code](https://github.com/jinyeying/night-enhancement) | — |

Image Harmonization – Image Harmonization
-----------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Harmonizer: Learning to Perform White-Box Image and Video Harmonization**](https://arxiv.org/abs/2207.01322) | [💻 Code](https://github.com/ZHKKKe/Harmonizer) | — |
| 2 | [**DCCF: Deep Comprehensive Color Filter Learning Framework for High-Resolution Image Harmonization**](https://arxiv.org/abs/2207.04788) | [💻 Code](https://github.com/rockeyben/DCCF) | — |
| 3 | [**Semantic-Guided Multi-Mask Image Harmonization**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3151_ECCV_2022_paper.php) | [💻 Code](https://github.com/XuqianRen/Semantic-guided-Multi-mask-Image-Harmonization) | — |
| 4 | [**Spatial-Separated Curve Rendering Network for Efficient and High-Resolution Image Harmonization**](https://arxiv.org/abs/2109.05750) | [💻 Code](https://github.com/stefanLeong/S2CRNet) | — |

Image Completion/Inpainting – image restoration
-----------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning Prior Feature and Attention Enhanced Image Inpainting**](https://arxiv.org/abs/2208.01837) | [💻 Code](https://github.com/ewrfcas/MAE-FAR) | — |
| 2 | [**Perceptual Artifacts Localization for Inpainting**](https://arxiv.org/abs/2208.03357) | [💻 Code](https://github.com/owenzlz/PAL4Inpaint) | — |
| 3 | [**High-Fidelity Image Inpainting with GAN Inversion**](https://arxiv.org/abs/2208.11850) | — | — |
| 4 | [**Unbiased Multi-Modality Guidance for Image Inpainting**](https://arxiv.org/abs/2208.11844) | — | — |
| 5 | [**Image Inpainting with Cascaded Modulation GAN and Object-Aware Training**](https://arxiv.org/abs/2203.11947) | [💻 Code](https://github.com/htzheng/CM-GAN-Inpainting) | — |
| 6 | [**Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5789_ECCV_2022_paper.php) | — | — |
| 7 | [**Diverse Image Inpainting with Normalizing Flow**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2814_ECCV_2022_paper.php) | — | — |
| 8 | [**Hourglass Attention Network for Image Inpainting**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3369_ECCV_2022_paper.php) | — | — |
| 9 | [**Perceptual Artifacts Localization for Inpainting**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2153_ECCV_2022_paper.php) | — | — |
| 10 | [**Don't Forget Me: Accurate Background Recovery for Text Removal via Modeling Local-Global Context**](https://arxiv.org/abs/2207.10273) | [💻 Code](https://github.com/lcy0604/CTRNet) | `Text Removal` |
| 11 | [**The Surprisingly Straightforward Scene Text Removal Method with Gated Attention and Region of Interest Generation: A Comprehensive Prominent Model Analysis**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4705_ECCV_2022_paper.php) | [💻 Code](https://github.com/naver/garnet) | `Text Removal` |

### Video Inpainting

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Error Compensation Framework for Flow-Guided Video Inpainting**](https://arxiv.org/abs/2207.10391) | — | — |
| 2 | [**Flow-Guided Transformer for Video Inpainting**](https://arxiv.org/abs/2208.06768) | [💻 Code](https://github.com/hitachinsk/FGT) | — |

Image Colorization – image colorization
---------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Eliminating Gradient Conflict in Reference-based Line-art Colorization**](https://arxiv.org/abs/2207.06095) | [💻 Code](https://github.com/kunkun0w0/SGA) | — |
| 2 | [**Bridging the Domain Gap towards Generalization in Automatic Colorization**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7304_ECCV_2022_paper.php) | [💻 Code](https://github.com/Lhyejin/DG-Colorization) | — |
| 3 | [**CT2: Colorization Transformer via Color Tokens**](https://ci.idm.pku.edu.cn/Weng_ECCV22b.pdf) | [💻 Code](https://github.com/shuchenweng/CT2) | — |
| 4 | [**PalGAN: Image Colorization with Palette Generative Adversarial Networks**](https://arxiv.org/abs/2210.11204) | [💻 Code](https://github.com/shepnerd/PalGAN) | — |
| 5 | [**BigColor: Colorization using a Generative Color Prior for Natural Images**](https://kimgeonung.github.io/assets/bigcolor/bigcolor_main.pdf) | [💻 Code](https://github.com/KIMGEONUNG/BigColor) | — |
| 6 | [**Semantic-Sparse Colorization Network for Deep Exemplar-Based Colorization**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/820_ECCV_2022_paper.php) | — | — |
| 7 | [**ColorFormer: Image Colorization via Color Memory Assisted Hybrid-Attention Transformer**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3385_ECCV_2022_paper.php) | — | — |
| 8 | [**L-CoDer: Language-Based Colorization with Color-Object Decoupling Transformer**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2424_ECCV_2022_paper.php) | — | — |
| 9 | [**Colorization for In Situ Marine Plankton Images**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6905_ECCV_2022_paper.php) | — | — |

Image Matting – image matting
-----------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**TransMatting: Enhancing Transparent Objects Matting with Transformers**](https://arxiv.org/abs/2208.03007) | [💻 Code](https://github.com/AceCHQ/TransMatting) | — |
| 2 | [**One-Trimap Video Matting**](https://arxiv.org/abs/2207.13353) | [💻 Code](https://github.com/Hongje/OTVM) | — |

Shadow Removal – shadow removal
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Style-Guided Shadow Removal**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5580_ECCV_2022_paper.php) | [💻 Code](https://github.com/jinwan1994/SG-ShadowNet) | — |

Image Compression – image compression
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Optimizing Image Compression via Joint Learning with Denoising**](https://arxiv.org/abs/2207.10869) | [💻 Code](https://github.com/felixcheng97/DenoiseCompression) | — |
| 2 | [**Implicit Neural Representations for Image Compression**](https://arxiv.org/abs/2112.04267) | [💻 Code](https://github.com/YannickStruempler/inr_based_compression) | — |
| 3 | [**Expanded Adaptive Scaling Normalization for End to End Image Compression**](https://arxiv.org/abs/2208.03049) | — | — |
| 4 | [**Content-Oriented Learned Image Compression**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7542_ECCV_2022_paper.php) | [💻 Code](https://github.com/lmijydyb/COLIC) | — |
| 5 | [**Contextformer: A Transformer with Spatio-Channel Attention for Context Modeling in Learned Image Compression**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6046_ECCV_2022_paper.php) | — | — |
| 6 | [**Content Adaptive Latents and Decoder for Neural Image Compression**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4016_ECCV_2022_paper.php) | — | — |

### Video Compression

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**AlphaVC: High-Performance and Efficient Learned Video Compression**](https://arxiv.org/abs/2207.14678) | — | — |
| 2 | [**CANF-VC: Conditional Augmented Normalizing Flows for Video Compression**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3904_ECCV_2022_paper.php) | [💻 Code](https://github.com/NYCU-MAPL/CANF-VC) | — |
| 3 | [**Neural Video Compression Using GANs for Detail Synthesis and Propagation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4802_ECCV_2022_paper.php) | — | — |

Image Quality Assessment – ​​image quality assessment
-----------------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling**](https://arxiv.org/abs/2207.02595) | [💻 Code](https://github.com/TimothyHTimothy/FAST-VQA) | — |
| 2 | [**Shift-tolerant Perceptual Similarity Metric**](https://arxiv.org/abs/2207.13686) | [💻 Code](https://github.com/abhijay9/ShiftTolerant-LPIPS/) | — |
| 3 | [**Telepresence Video Quality Assessment**](https://arxiv.org/abs/2207.09956) | — | — |
| 4 | [**A Perceptual Quality Metric for Video Frame Interpolation**](https://arxiv.org/abs/2210.01879) | [💻 Code](https://github.com/hqqxyy/VFIPS) | — |

Relighting/Delighting
---------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Deep Portrait Delighting**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4581_ECCV_2022_paper.php) | — | — |
| 2 | [**Geometry-Aware Single-Image Full-Body Human Relighting**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4385_ECCV_2022_paper.php) | — | — |
| 3 | [**NeRF for Outdoor Scene Relighting**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4998_ECCV_2022_paper.php) | — | — |
| 4 | [**Physically-Based Editing of Indoor Scene Lighting from a Single Image**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1276_ECCV_2022_paper.php) | — | — |

Style Transfer – style transfer
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer**](https://arxiv.org/abs/2207.04808) | [💻 Code](https://github.com/JarrentWu1031/CCPL) | — |
| 2 | [**Image-Based CLIP-Guided Essence Transfer**](https://arxiv.org/abs/2110.12427) | [💻 Code](https://github.com/hila-chefer/TargetCLIP) | — |
| 3 | [**Learning Graph Neural Networks for Image Style Transfer**](https://arxiv.org/abs/2207.11681) | — | — |
| 4 | [**WISE: Whitebox Image Stylization by Example-based Learning**](https://arxiv.org/abs/2207.14606) | [💻 Code](https://github.com/winfried-loetzsch/wise) | — |
| 5 | [**Language-Driven Artistic Style Transfer**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6627_ECCV_2022_paper.php) | — | — |
| 6 | [**MoDA: Map Style Transfer for Self-Supervised Domain Adaptation of Embodied Agents**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1762_ECCV_2022_paper.php) | — | — |
| 7 | [**JoJoGAN: One Shot Face Stylization**](https://arxiv.org/abs/2112.11641) | [💻 Code](https://github.com/mchong6/JoJoGAN) | — |
| 8 | [**EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer**](https://arxiv.org/abs/2207.09840) | [💻 Code](https://github.com/Chenyu-Yang-2000/EleGANt) | `Makeup Transfer` |
| 9 | [**RamGAN: Region Attentive Morphing GAN for Region-Level Makeup Transfer**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/803_ECCV_2022_paper.php) | — | `Makeup Transfer` |

Image Editing – image editing
-----------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Context-Consistent Semantic Image Editing with Style-Preserved Modulation**](https://arxiv.org/abs/2207.06252) | [💻 Code](https://github.com/WuyangLuo/SPMPGAN) | — |
| 2 | [**GAN with Multivariate Disentangling for Controllable Hair Editing**](https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/main/database/CtrlHair/CtrlHair.pdf) | [💻 Code](https://github.com/XuyangGuo/CtrlHair) | — |
| 3 | [**Paint2Pix: Interactive Painting based Progressive Image Synthesis and Editing**](https://arxiv.org/abs/2208.08092) | [💻 Code](https://github.com/1jsingh/paint2pix) | — |
| 4 | [**High-fidelity GAN Inversion with Padding Space**](https://arxiv.org/abs/2203.11105) | [💻 Code](https://github.com/EzioBy/padinv) | — |
| 5 | [**Text2LIVE: Text-Driven Layered Image and Video Editing**](https://arxiv.org/abs/2204.02491) | [💻 Code](https://github.com/omerbt/Text2LIVE) | — |
| 6 | [**IntereStyle: Encoding an Interest Region for Robust StyleGAN Inversion**](https://arxiv.org/abs/2209.10811) | — | — |
| 7 | [**Style Your Hair: Latent Optimization for Pose-Invariant Hairstyle Transfer via Local-Style-Aware Hair Alignment**](https://arxiv.org/abs/2208.07765) | [💻 Code](https://github.com/Taeu/Style-Your-Hair) | — |
| 8 | [**HairNet: Hairstyle Transfer with Pose Changes**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5227_ECCV_2022_paper.php) | — | — |
| 9 | [**End-to-End Visual Editing with a Generatively Pre-trained Artist**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/841_ECCV_2022_paper.php) | — | — |
| 10 | [**The Anatomy of Video Editing: A Dataset and Benchmark Suite for AI-Assisted Video Editing**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4736_ECCV_2022_paper.php) | — | — |
| 11 | [**Scraping Textures from Natural Images for Synthesis and Editing**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2180_ECCV_2022_paper.php) | — | — |
| 12 | [**VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language Guidance**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/8048_ECCV_2022_paper.php) | — | — |
| 13 | [**Editing Out-of-Domain GAN Inversion via Differential Activations**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5504_ECCV_2022_paper.php) | [💻 Code](https://github.com/HaoruiSong622/Editing-Out-of-Domain) | — |
| 14 | [**ChunkyGAN: Real Image Inversion via Segments**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5092_ECCV_2022_paper.php) | — | — |
| 15 | [**FairStyle: Debiasing StyleGAN2 with Style Channel Manipulations**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7746_ECCV_2022_paper.php) | [💻 Code](https://github.com/catlab-team/fairstyle) | — |
| 16 | [**A Style-Based GAN Encoder for High Fidelity Reconstruction of Images and Videos**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2740_ECCV_2022_paper.php) | [💻 Code](https://github.com/InterDigitalInc/FeatureStyleEncoder) | — |
| 17 | [**Rayleigh EigenDirections (REDs): Nonlinear GAN latent space traversals for multidimensional features**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7277_ECCV_2022_paper.php) | — | — |

Image Generation/Synthesis / Image-to-Image Translation – Image Generation/Synthesis/Translation
------------------------------------------------------------------------------------------------

### Text-to-Image / Text Guided / Multi-Modal

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**TIPS: Text-Induced Pose Synthesis**](https://arxiv.org/abs/2207.11718) | [💻 Code](https://github.com/prasunroy/tips) | — |
| 2 | [**TISE: A Toolbox for Text-to-Image Synthesis Evaluation**](https://arxiv.org/abs/2112.01398) | [💻 Code](https://github.com/VinAIResearch/tise-toolbox) | — |
| 3 | [**Learning Visual Styles from Audio-Visual Associations**](https://arxiv.org/abs/2205.05072) | [💻 Code](https://github.com/Tinglok/avstyle) | — |
| 4 | [**Multimodal Conditional Image Synthesis with Product-of-Experts GANs**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3539_ECCV_2022_paper.php) | [🌐 Project](https://deepimagination.cc/PoE-GAN/) | — |
| 5 | [**NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5422_ECCV_2022_paper.php) | — | — |
| 6 | [**Make-a-Scene: Scene-Based Text-to-Image Generation with Human Priors**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/993_ECCV_2022_paper.php) | — | — |
| 7 | [**Trace Controlled Text to Image Generation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1894_ECCV_2022_paper.php) | — | — |
| 8 | [**Audio-Driven Stylized Gesture Generation with Flow-Based Model**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3948_ECCV_2022_paper.php) | — | — |
| 9 | [**No Token Left Behind: Explainability-Aided Image Classification and Generation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2764_ECCV_2022_paper.php) | — | — |

### Image-to-Image / Image Guided

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**End-to-end Graph-constrained Vectorized Floorplan Generation with Panoptic Refinement**](https://arxiv.org/abs/2207.13268) | — | — |
| 2 | [**ManiFest: Manifold Deformation for Few-shot Image Translation**](https://arxiv.org/abs/2111.13681) | [💻 Code](https://github.com/cv-rits/ManiFest) | — |
| 3 | [**VecGAN: Image-to-Image Translation with Interpretable Latent Directions**](https://arxiv.org/abs/2207.03411) | — | — |
| 4 | [**DynaST: Dynamic Sparse Transformer for Exemplar-Guided Image Generation**](https://arxiv.org/abs/2207.06124) | [💻 Code](https://github.com/Huage001/DynaST) | — |
| 5 | [**Cross Attention Based Style Distribution for Controllable Person Image Synthesis**](https://arxiv.org/abs/2208.00712) | [💻 Code](https://github.com/xyzhouo/CASD) | — |
| 6 | [**Vector Quantized Image-to-Image Translation**](https://arxiv.org/abs/2207.13286) | [💻 Code](https://github.com/cyj407/VQ-I2I) | — |
| 7 | [**URUST: Ultra-high-resolution unpaired stain transformation via Kernelized Instance Normalization**](https://arxiv.org/abs/2208.10730) | [💻 Code](https://github.com/Kaminyou/URUST) | — |
| 8 | [**General Object Pose Transformation Network from Unpaired Data**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5972_ECCV_2022_paper.php) | [💻 Code](https://github.com/suyukun666/UFO-PT) | — |
| 9 | [**Unpaired Image Translation via Vector Symbolic Architectures**](https://arxiv.org/abs/2209.02686) | [💻 Code](https://github.com/facebookresearch/vsait) | — |
| 10 | [**Supervised Attribute Information Removal and Reconstruction for Image Manipulation**](https://arxiv.org/abs/2207.06555) | [💻 Code](https://github.com/NannanLi999/AIRR) | — |
| 11 | [**Bi-Level Feature Alignment for Versatile Image Translation and Manipulation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3912_ECCV_2022_paper.php) | — | — |
| 12 | [**Multi-Curve Translator for High-Resolution Photorealistic Image Translation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1278_ECCV_2022_paper.php) | — | — |
| 13 | [**CoGS: Controllable Generation and Search from Sketch and Style**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5160_ECCV_2022_paper.php) | — | — |
| 14 | [**AgeTransGAN for Facial Age Transformation with Rectified Performance Metrics**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1344_ECCV_2022_paper.php) | [💻 Code](https://github.com/AvLab-CV/AgeTransGAN) | — |

### Others for image generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**StyleLight: HDR Panorama Generation for Lighting Estimation and Editing**](https://arxiv.org/abs/2207.14811) | [💻 Code](https://github.com/Wanggcong/StyleLight) | — |
| 2 | [**Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling**](https://arxiv.org/abs/2207.02196) | [💻 Code](https://github.com/fudan-zvg/PDS) | — |
| 3 | [**GAN Cocktail: mixing GANs without dataset access**](https://arxiv.org/abs/2106.03847) | [💻 Code](https://github.com/omriav/GAN-cocktail) | — |
| 4 | [**Compositional Visual Generation with Composable Diffusion Models**](https://arxiv.org/abs/2206.01714) | [💻 Code](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch) | — |
| 5 | [**Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation**](https://arxiv.org/abs/2112.02450) | [💻 Code](https://github.com/dzld00/Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation) | — |
| 6 | [**StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pretrained StyleGAN**](https://arxiv.org/abs/2203.04036) | [💻 Code](https://github.com/FeiiYin/StyleHEAT) | — |
| 7 | [**WaveGAN: An Frequency-aware GAN for High-Fidelity Few-shot Image Generation**](https://arxiv.org/abs/2207.07288) | [💻 Code](https://github.com/kobeshegu/ECCV2022_WaveGAN) | — |
| 8 | [**FakeCLR: Exploring Contrastive Learning for Solving Latent Discontinuity in Data-Efficient GANs**](https://arxiv.org/abs/2207.08630) | [💻 Code](https://github.com/iceli1007/FakeCLR) | — |
| 9 | [**Auto-regressive Image Synthesis with Integrated Quantization**](https://arxiv.org/abs/2207.10776) | [💻 Code](https://github.com/fnzhan/IQ-VAE) | — |
| 10 | [**PixelFolder: An Efficient Progressive Pixel Synthesis Network for Image Generation**](https://arxiv.org/abs/2204.00833) | [💻 Code](https://github.com/BlingHe/PixelFolder) | — |
| 11 | [**DeltaGAN: Towards Diverse Few-shot Image Generation with Sample-Specific Delta**](https://arxiv.org/abs/2207.10271) | [💻 Code](https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation) | — |
| 12 | [**Generator Knows What Discriminator Should Learn in Unconditional GANs**](https://arxiv.org/abs/2207.13320) | [💻 Code](https://github.com/naver-ai/GGDR) | — |
| 13 | [**Hierarchical Semantic Regularization of Latent Spaces in StyleGANs**](https://arxiv.org/abs/2208.03764) | [💻 Code](https://drive.google.com/file/d/1gzHTYTgGBUlDWyN_Z3ORofisQrHChg_n/view) | — |
| 14 | [**FurryGAN: High Quality Foreground-aware Image Synthesis**](https://arxiv.org/abs/2208.10422) | [🌐 Project](https://jeongminb.github.io/FurryGAN/) | — |
| 15 | [**Improving GANs for Long-Tailed Data through Group Spectral Regularization**](https://arxiv.org/abs/2208.09932) | [💻 Code](https://drive.google.com/file/d/1aG48i04Q8mOmD968PAgwEvPsw1zcS4Gk/view) | — |
| 16 | [**Exploring Gradient-based Multi-directional Controls in GANs**](https://arxiv.org/abs/2209.00698) | [💻 Code](https://github.com/zikuncshelly/GradCtrl) | — |
| 17 | [**Improved Masked Image Generation with Token-Critic**](https://arxiv.org/abs/2209.04439) | — | — |
| 18 | [**Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation**](https://arxiv.org/abs/2209.05968) | [🌐 Project](https://eadcat.github.io/WSSN) | — |
| 19 | [**Any-Resolution Training for High-Resolution Image Synthesis**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3693_ECCV_2022_paper.php) | [💻 Code](https://github.com/chail/anyres-gan) | — |
| 20 | [**BIPS: Bi-modal Indoor Panorama Synthesis via Residual Depth-Aided Adversarial Learning**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4327_ECCV_2022_paper.php) | [💻 Code](https://github.com/chang9711/BIPS) | — |
| 21 | [**Few-Shot Image Generation with Mixup-Based Distance Learning**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2709_ECCV_2022_paper.php) | [💻 Code](https://github.com/reyllama/mixdl) | — |
| 22 | [**StyleGAN-Human: A Data-Centric Odyssey of Human Generation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3366_ECCV_2022_paper.php) | [💻 Code](https://github.com/stylegan-human/StyleGAN-Human) | — |
| 23 | [**StyleFace: Towards Identity-Disentangled Face Generation on Megapixels**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4255_ECCV_2022_paper.php) | — | — |
| 24 | [**Contrastive Learning for Diverse Disentangled Foreground Generation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4323_ECCV_2022_paper.php) | — | — |
| 25 | [**BLT: Bidirectional Layout Transformer for Controllable Layout Generation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7035_ECCV_2022_paper.php) | [💻 Code](https://github.com/google-research/google-research/tree/master/layout-blt) | — |
| 26 | [**Entropy-Driven Sampling and Training Scheme for Conditional Diffusion Generation**](https://arxiv.org/abs/2206.11474) | [💻 Code](https://github.com/ZGCTroy/ED-DPM) | — |
| 27 | [**Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5081_ECCV_2022_paper.php) | — | — |
| 28 | [**DuelGAN: A Duel between Two Discriminators Stabilizes the GAN Training**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7143_ECCV_2022_paper.php) | [💻 Code](https://github.com/UCSC-REAL/DuelGAN) | — |

### Video Generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer**](https://arxiv.org/abs/2204.03638) | [💻 Code](https://github.com/SongweiGe/TATS) | — |
| 2 | [**Controllable Video Generation through Global and Local Motion Dynamics**](https://arxiv.org/abs/2204.06558) | [💻 Code](https://github.com/Araachie/glass) | — |
| 3 | [**Fast-Vid2Vid: Spatial-Temporal Compression for Video-to-Video Synthesis**](https://arxiv.org/abs/2207.05049) | [💻 Code](https://github.com/fast-vid2vid/fast-vid2vid) | — |
| 4 | [**Synthesizing Light Field Video from Monocular Video**](https://arxiv.org/abs/2207.10357) | [💻 Code](https://github.com/ShrisudhanG/Synthesizing-Light-Field-Video-from-Monocular-Video) | — |
| 5 | [**StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation**](https://arxiv.org/abs/2209.06192) | [💻 Code](https://github.com/adymaharana/storydalle) | — |
| 6 | **Motion Transformer for Unsupervised Image Animation** | [💻 Code](https://github.com/JialeTao/MoTrans) | — |
| 7 | [**Sound-Guided Semantic Video Generation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5584_ECCV_2022_paper.php) | [💻 Code](https://github.com/anonymous5584/sound-guided-semantic-video-generation) | — |
| 8 | [**Layered Controllable Video Generation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4847_ECCV_2022_paper.php) | — | — |
| 9 | [**Diverse Generation from a Single Video Made Possible**](https://arxiv.org/abs/2109.08591) | [💻 Code](https://github.com/nivha/single_video_generation) | — |
| 10 | [**Semantic-Aware Implicit Neural Audio-Driven Video Portrait Generation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/631_ECCV_2022_paper.php) | [💻 Code](https://github.com/alvinliu0/SSP-NeRF) | — |
| 11 | [**EAGAN: Efficient Two-Stage Evolutionary Architecture Search for GANs**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3419_ECCV_2022_paper.php) | [💻 Code](https://github.com/marsggbo/EAGAN) | — |
| 12 | [**BlobGAN: Spatially Disentangled Scene Representations**](https://arxiv.org/abs/2205.02837) | [💻 Code](https://github.com/dave-epstein/blobgan) | — |

Others
------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning Local Implicit Fourier Representation for Image Warping**](https://ipl.dgist.ac.kr/LTEW.pdf) | [💻 Code](https://github.com/jaewon-lee-b/ltew) | `Image Warping` |
| 2 | [**Dress Code: High-Resolution Multi-Category Virtual Try-On**](https://arxiv.org/abs/2204.08532) | [💻 Code](https://github.com/aimagelab/dress-code) | `Virtual Try-On` |
| 3 | [**High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions**](https://arxiv.org/abs/2206.14180) | [💻 Code](https://github.com/sangyun884/HR-VITON) | `Virtual Try-On` |
| 4 | [**Single Stage Virtual Try-on via Deformable Attention Flows**](https://arxiv.org/abs/2207.09161) | — | `Virtual Try-On` |
| 5 | [**Outpainting by Queries**](https://arxiv.org/abs/2207.05312) | [💻 Code](https://github.com/Kaiseem/QueryOTR) | `Outpainting` |
| 6 | [**Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal**](https://arxiv.org/abs/2207.08178) | [💻 Code](https://github.com/thinwayliu/Watermark-Vaccine) | `Watermark Protection` |
| 7 | [**Efficient Meta-Tuning for Content-aware Neural Video Delivery**](https://arxiv.org/abs/2207.09691) | [💻 Code](https://github.com/Neural-video-delivery/EMT-Pytorch-ECCV2022) | `Video Delivery` |
| 8 | [**Human-centric Image Cropping with Partition-aware and Content-preserving Features**](https://arxiv.org/abs/2207.10269) | [💻 Code](https://github.com/bcmi/Human-Centric-Image-Cropping) | — |
| 9 | [**CelebV-HQ: A Large-Scale Video Facial Attributes Dataset**](https://arxiv.org/abs/2207.12393) | [💻 Code](https://github.com/CelebV-HQ/CelebV-HQ) | `Dataset` |
| 10 | [**Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis**](https://arxiv.org/abs/2207.11770) | [💻 Code](https://github.com/sstzal/DFRF) | `Talking Head Synthesis` |
| 11 | [**Responsive Listening Head Generation: A Benchmark Dataset and Baseline**](https://arxiv.org/abs/2112.13548) | [💻 Code](https://github.com/dc3ea9f/vico_challenge_baseline) | — |
| 12 | [**Contrastive Monotonic Pixel-Level Modulation**](https://arxiv.org/abs/2207.11517) | [💻 Code](https://github.com/lukun199/MonoPix) | — |
| 13 | [**AutoTransition: Learning to Recommend Video Transition Effects**](https://arxiv.org/abs/2207.13479) | [💻 Code](https://github.com/acherstyx/AutoTransition) | — |
| 14 | [**Bringing Rolling Shutter Images Alive with Dual Reversed Distortion**](https://arxiv.org/abs/2203.06451) | [💻 Code](https://github.com/zzh-tech/Dual-Reversed-RS) | — |
| 15 | [**Learning Object Placement via Dual-path Graph Completion**](https://arxiv.org/abs/2207.11464) | [💻 Code](https://github.com/bcmi/GracoNet-Object-Placement) | — |
| 16 | [**DeepMCBM: A Deep Moving-camera Background Model**](https://arxiv.org/abs/2209.07923) | [💻 Code](https://github.com/BGU-CS-VIL/DeepMCBM) | — |
| 17 | [**Mind the Gap in Distilling StyleGANs**](https://arxiv.org/abs/2208.08840) | [💻 Code](https://github.com/xuguodong03/StyleKD) | — |
| 18 | [**StyleSwap: Style-Based Generator Empowers Robust Face Swapping**](https://arxiv.org/abs/2209.13514) | [💻 Code](https://github.com/Seanseattle/StyleSwap) | `Face Swapping` |
| 19 | [**Geometric Representation Learning for Document Image Rectification**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1698_ECCV_2022_paper.php) | [💻 Code](https://github.com/fh2019ustc/DocGeoNet) | `Document Image Rectification` |
| 20 | [**Studying Bias in GANs through the Lens of Race**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2581_ECCV_2022_paper.php) | — | `Racial Bias` |
| 21 | [**On the Robustness of Quality Measures for GANs**](https://arxiv.org/abs/2201.13019) | [💻 Code](https://github.com/MotasemAlfarra/R-FID-Robustness-of-Quality-Measures-for-GANs) | — |
| 22 | [**TREND: Truncated Generalized Normal Density Estimation of Inception Embeddings for GAN Evaluation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3604_ECCV_2022_paper.php) | — | `GAN Evaluation` |

AAAI2022-Low-Level-Vision
=========================

Image Restoration 
------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Unsupervised Underwater Image Restoration: From a Homology Perspective**](https://aaai-2022.virtualchair.net/poster_aaai2078) | — | `Underwater Image Restoration` |
| 2 | [**Panini-Net: GAN Prior based Degradation-Aware Feature Interpolation for Face Restoration**](https://aaai-2022.virtualchair.net/poster_aaai4252) | [💻 Code](https://github.com/wyhuai/Panini-Net) | `Face Restoration` |

### Burst Restoration

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Zero-Shot Multi-Frame Image Restoration with Pre-Trained Siamese Transformers**](https://aaai-2022.virtualchair.net/poster_aaai7488) | [💻 Code](https://github.com/laulampaul/siamtrans) | — |

### Video Restoration

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Transcoded Video Restoration by Temporal Spatial Auxiliary Network**](https://aaai-2022.virtualchair.net/poster_aaai12302) | — | `Transcoded Video Restoration` |

Super Resolution 
-----------------------

### Image Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**SCSNet: An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-Resolution**](https://aaai-2022.virtualchair.net/poster_aaai528) | — | — |
| 2 | [**Efficient Non-Local Contrastive Attention for Image Super-Resolution**](https://arxiv.org/abs/2201.03794) | [💻 Code](https://github.com/Zj-BinXia/ENLCA) | — |
| 3 | [**Best-Buddy GANs for Highly Detailed Image Super-Resolution**](https://aaai-2022.virtualchair.net/poster_aaai137) | — | `GAN` |
| 4 | [**Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution**](https://aaai-2022.virtualchair.net/poster_aaai1194) | — | `Text SR` |
| 5 | [**Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-Based Super-Resolution**](https://aaai-2022.virtualchair.net/poster_aaai390) | [💻 Code](https://github.com/Zj-BinXia/AMSA) | `Reference-Based SR` |
| 6 | [**Detail-Preserving Transformer for Light Field Image Super-Resolution**](https://aaai-2022.virtualchair.net/poster_aaai3550) | — | `Light Field` |

Denoising – denoising
---------------------

### Image Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Generative Adaptive Convolutions for Real-World Noisy Image Denoising**](https://aaai-2022.virtualchair.net/poster_aaai4230) | — | — |

### Video Denoising

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ReMoNet: Recurrent Multi-Output Network for Efficient Video Denoising**](https://aaai-2022.virtualchair.net/poster_aaai6295) | — | — |

Deblurring – Deblurring
-----------------------

### Video Deblurring

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Deep Recurrent Neural Network with Multi-Scale Bi-Directional Propagation for Video Deblurring**](https://aaai-2022.virtualchair.net/poster_aaai3124) | — | — |

Deraining – deraining
---------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Online-Updated High-Order Collaborative Networks for Single Image Deraining**](https://aaai-2022.virtualchair.net/poster_aaai6295) | — | — |
| 2 | [**Close the Loop: A Unified Bottom-up and Top-down Paradigm for Joint Image Deraining and Segmentation**](https://aaai-2022.virtualchair.net/poster_aaai678) | — | `Joint Image Deraining and Segmentation` |

Dehazing – to fog
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Uncertainty-Driven Dehazing Network**](https://aaai-2022.virtualchair.net/poster_aaai2838) | — | — |

Demosaicing – Demosaicing
-------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Deep Spatial Adaptive Network for Real Image Demosaicing**](https://aaai-2022.virtualchair.net/poster_aaai2170) | — | — |

HDR Imaging / Multi-Exposure Image Fusion – HDR image generation / multi-exposure image fusion
----------------------------------------------------------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework Using Self-Supervised Multi-Task Learning**](https://arxiv.org/abs/2112.01030) | [💻 Code](https://github.com/miccaiif/TransMEF) | — |

Image Enhancement – ​​image enhancement
---------------------------------------

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Low-Light Image Enhancement with Normalizing Flow**](https://arxiv.org/abs/2109.05923) | [💻 Code](https://github.com/wyf0912/LLFlow) | — |
| 2 | [**Degrade is Upgrade: Learning Degradation for Low-light Image Enhancement**](https://aaai-2022.virtualchair.net/poster_aaai841) | — | — |
| 3 | [**Semantically Contrastive Learning for Low-Light Image Enhancement**](https://aaai-2022.virtualchair.net/poster_aaai4218) | — | `contrastive learning` |

Image Matting – image matting
-----------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**MODNet: Trimap-Free Portrait Matting in Real Time**](https://arxiv.org/abs/2011.11961) | [💻 Code](https://github.com/ZHKKKe/MODNet) | — |

Shadow Removal – shadow removal
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Efficient Model-Driven Network for Shadow Removal**](https://aaai-2022.virtualchair.net/poster_aaai196) | — | — |

Image Compression – image compression
-------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Towards End-to-End Image Compression and Analysis with Transformers**](https://arxiv.org/abs/2112.09300) | [💻 Code](https://github.com/BYchao100/Towards-Image-Compression-and-Analysis-with-Transformers) | `Transformer` |
| 2 | [**OoDHDR-Codec: Out-of-Distribution Generalization for HDR Image Compression**](https://aaai-2022.virtualchair.net/poster_aaai8610) | — | — |
| 3 | [**Two-Stage Octave Residual Network for End-to-End Image Compression**](https://aaai-2022.virtualchair.net/poster_aaai4043) | — | — |

Image Quality Assessment – ​​image quality assessment
-----------------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Content-Variant Reference Image Quality Assessment via Knowledge Distillation**](https://aaai-2022.virtualchair.net/poster_aaai1344) | — | — |
| 2 | [**Perceptual Quality Assessment of Omnidirectional Images**](https://aaai-2022.virtualchair.net/poster_aaai4008) | — | `Omnidirectional Images` |

Style Transfer – style transfer
-------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization**](https://arxiv.org/abs/2103.11784) | [💻 Code](https://github.com/czczup/URST) | — |
| 2 | [**Deep Translation Prior: Test-Time Training for Photorealistic Style Transfer**](https://aaai-2022.virtualchair.net/poster_aaai1958) | — | — |

Image Editing – image editing
-----------------------------

Image Generation/Synthesis / Image-to-Image Translation – Image Generation/Synthesis/Translation
------------------------------------------------------------------------------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**SSAT: A Symmetric Semantic-Aware Transformer Network for Makeup Transfer and Removal**](https://aaai-2022.virtualchair.net/poster_aaai934) | [💻 Code](https://github.com/Snowfallingplum/SSAT) | `Makeup Transfer and Removal` |
| 2 | [**Assessing a Single Image in Reference-Guided Image Synthesis**](https://aaai-2022.virtualchair.net/poster_aaai1241) | — | — |
| 3 | [**Interactive Image Generation with Natural-Language Feedback**](https://aaai-2022.virtualchair.net/poster_aaai7081) | — | — |
| 4 | [**PetsGAN: Rethinking Priors for Single Image Generation**](https://aaai-2022.virtualchair.net/poster_aaai2865) | — | — |
| 5 | [**Pose Guided Image Generation from Misaligned Sources via Residual Flow Based Correction**](https://aaai-2022.virtualchair.net/poster_aaai4099) | — | — |
| 6 | [**Hierarchical Image Generation via Transformer-Based Sequential Patch Selection**](https://aaai-2022.virtualchair.net/poster_aaai557) | — | — |
| 7 | [**Style-Guided and Disentangled Representation for Robust Image-to-Image Translation**](https://aaai-2022.virtualchair.net/poster_aaai3727) | — | — |
| 8 | [**OA-FSUI2IT: A Novel Few-Shot Cross Domain Object Detection Framework with Object-Aware Few-shot Unsupervised Image-to-Image Translation**](https://aaai-2022.virtualchair.net/poster_aaai2213) | [💻 Code](https://github.com/emdata-ailab/FSCD-Det) | `Image-to-Image Translation used for Object Detection` |

### Video Generation

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Learning Temporally and Semantically Consistent Unpaired Video-to-Video Translation through Pseudo-Supervision from Synthetic Optical Flow**](https://aaai-2022.virtualchair.net/poster_aaai4610) | [💻 Code](https://github.com/wangkaihong/Unsup_Recycle_GAN) | — |

ECCV2024-Low-Level-Vision
=========================

Image Restoration
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**InstructIR: High-Quality Image Restoration Following Human Instructions**](https://arxiv.org/abs/2401.16468) | [💻 Code](https://github.com/mv-lab/InstructIR) | `Instruction-Based, All-in-One Restoration` |
| 2 | [**Restore Anything with Masks: Leveraging Mask Image Modeling for Blind All-in-One Image Restoration**](https://arxiv.org/abs/2312.12529) | [💻 Code](https://github.com/Dragonisss/RAM) | `All-in-One Restoration, Masked Modeling` |
| 3 | [**Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration**](https://arxiv.org/abs/2308.10246) | — | `All-in-One Restoration, Degradation Perception` |

### Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**AdcSR: Towards Real-World Blind Super-Resolution via Adaptive Degradation-aware Contrastive Learning**](https://arxiv.org/abs/2408.05407) | — | `Real-World SR, Contrastive Learning` |
| 2 | [**Arbitrary-Scale Video Super-Resolution with Structural and Textural Priors**](https://arxiv.org/abs/2407.09919) | [💻 Code](https://github.com/shangwei5/ST-AVSR) | `Video SR, Arbitrary Scale` |

Denoising
---------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Hierarchical Intra-frame Motion Learning for Video Frame Interpolation**](https://arxiv.org/abs/2407.09948) | — | `Denoising, Hierarchical` |
| 2 | [**Blind Image Denoising via Fast Diffusion Inversion**](https://arxiv.org/abs/2307.07179) | — | `Diffusion, Blind Denoising` |

Deblurring
----------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Motion Blur Decomposition with Cross-shutter Guidance**](https://arxiv.org/abs/2404.01120) | — | `Deblurring, Cross-shutter` |
| 2 | [**LoFormer: Local Frequency Transformer for Image Deblurring**](https://arxiv.org/abs/2407.16993) | [💻 Code](https://github.com/INVOKERer/LoFormer) | `Transformer, Deblurring, Frequency Domain` |

Dehazing
--------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**FreqMamba: Viewing Mamba from a Frequency Perspective for Image Deraining**](https://arxiv.org/abs/2404.09476) | — | `Mamba, Frequency Domain, Deraining` |
| 2 | [**OKNet: Omni-Knowledge Network for All-In-One Image Restoration**](https://arxiv.org/abs/2408.02818) | — | `All-in-One, Knowledge Distillation` |

Image Enhancement
-----------------

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**NeRCo: Normalizing Implicit Representational Coherence for Unsupervised Low-light Image Enhancement**](https://arxiv.org/abs/2310.03031) | [💻 Code](https://github.com/wenci0024/NeRCo) | `Unsupervised, Low-Light, Implicit Representation` |
| 2 | [**RCTNet: Real-time Color Transfer Between Images**](https://arxiv.org/abs/2407.11721) | — | `Color Transfer, Real-time` |

Frame Interpolation
-------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Generalizable Implicit Neural Representation for Ultra-High-Definition Frame Interpolation**](https://arxiv.org/abs/2407.13827) | — | `Frame Interpolation, Implicit Neural Representation, Ultra-HD` |

Image Generation/Synthesis
--------------------------

### Text-to-Image / Diffusion

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**](https://arxiv.org/abs/2307.01952) | [💻 Code](https://github.com/Stability-AI/generative-models) | `Latent Diffusion, High-Resolution` |
| 2 | [**Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3)**](https://arxiv.org/abs/2403.03206) | — | `Rectified Flow, High-Resolution, Diffusion Transformer` |
| 3 | [**CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer**](https://arxiv.org/abs/2408.06072) | [💻 Code](https://github.com/THUDM/CogVideo) | `Text-to-Video, Diffusion Transformer` |

Image Quality Assessment
------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Compare2Score: In-Context Learning Makes LLMs Strong Zero-Shot Image Quality Evaluator**](https://arxiv.org/abs/2401.16462) | — | `IQA, LLM, Zero-Shot` |
| 2 | [**TOPIQ: A Top-down Perspective from Semantics to Distortions for Image Quality Assessment**](https://arxiv.org/abs/2308.03060) | [💻 Code](https://github.com/chaofengc/IQA-PyTorch) | `IQA, Top-down, Semantics` |

---

NeurIPS2023-Low-Level-Vision
=============================

Image Restoration
-----------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**DiffIR: Efficient Diffusion Model for Image Restoration**](https://arxiv.org/abs/2303.09472) | [💻 Code](https://github.com/Zj-BinXia/DiffIR) | `Diffusion, Image Restoration` |
| 2 | [**Improving Image Restoration through Removing Degradations in Textual Representations**](https://arxiv.org/abs/2312.17334) | — | `Text-Guided, All-in-One Restoration` |

### Super Resolution

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting**](https://arxiv.org/abs/2307.12348) | [💻 Code](https://github.com/zsyOAOA/ResShift) | `Diffusion SR, Residual Shift, Efficient` |
| 2 | [**Iterative Token Evaluation and Refinement for Real-World Super-Resolution**](https://arxiv.org/abs/2312.05616) | [💻 Code](https://github.com/chaofengc/ITER) | `Real-World SR, Token Refinement` |

Image Enhancement
-----------------

### Low-Light Image Enhancement

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Implicit Neural Representation for Cooperative Low-light Image Enhancement**](https://arxiv.org/abs/2303.11722) | — | `Implicit Neural Representation, Low-Light` |
| 2 | [**Fourier-based Augmentation with Applications to Domain Generalization**](https://arxiv.org/abs/2309.09145) | — | `Fourier, Domain Generalization, Augmentation` |

Image Generation/Synthesis
--------------------------

### Diffusion Models

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**Consistency Models**](https://arxiv.org/abs/2303.01469) | [💻 Code](https://github.com/openai/consistency_models) | `Consistency, Fast Sampling, Score Matching` |
| 2 | [**Diffusion Model for Dense Prediction**](https://arxiv.org/abs/2303.04309) | — | `Dense Prediction, Diffusion` |
| 3 | [**Imagic: Text-Based Real Image Editing with Diffusion Models**](https://arxiv.org/abs/2210.09276) | — | `Image Editing, Diffusion` |
| 4 | [**Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation**](https://arxiv.org/abs/2404.04057) | [💻 Code](https://github.com/mingyuanzhou/SiD) | `Distillation, One-Step, Diffusion` |

Image Quality Assessment
------------------------

| # | Paper | Code | Tags |
|:---:|:---|:---:|:---|
| 1 | [**UNIQUE: Uncertainty-aware blind image quality assessment in the laboratory and wild**](https://arxiv.org/abs/2108.05797) | [💻 Code](https://github.com/zwx8981/UNIQUE) | `IQA, Uncertainty, Blind` |

---

Reference
========

[What are low-level and high-level tasks\_low-level tasks\_WTHunt's Blog-CSDN Blog](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://blog.csdn.net/qq_20880415/article/details/117225213 "What are low-level and high-level tasks_low-level tasks_WTHunt's Blog-CSDN Blog")

[What is the prospect of low-level vision in the CV field? – Zhihu (zhihu.com)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.zhihu.com/question/467432767 "What is the prospect of low-level vision in the CV field?  - Zhihu (zhihu.com)")

[GitHub – DarrenPan/Awesome-CVPR2023-Low-Level-Vision: A Collection of Papers and Codes in CVPR2023/2022 about low level vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-CVPR2023-Low-Level-Vision "GitHub - DarrenPan/Awesome-CVPR2023-Low-Level-Vision: A Collection of Papers and Codes in CVPR2023/2022 about low level vision")

*   [Awesome-CVPR2024-Low-Level-Vision](https://github.com/DarrenPan/Awesome-CVPR2024-Low-Level-Vision)
*   [Awesome-ICCV2023-Low-Level-Vision](https://github.com/DarrenPan/Awesome-ICCV2023-Low-Level-Vision)
*   [Awesome-CVPR2023-Low-Level-Vision](https://github.com/DarrenPan/Awesome-CVPR2023-Low-Level-Vision)
*   [Awesome-CVPR2022-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-CVPR2023-Low-Level-Vision/blob/main/CVPR2022-Low-Level-Vision.md "Awesome-CVPR2022-Low-Level-Vision")
*   [Awesome-ECCV2022-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-ECCV2022-Low-Level-Vision "Awesome-ECCV2022-Low-Level-Vision")
*   [Awesome-AAAI2022-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-AAAI2022-Low-Level-Vision "Awesome-AAAI2022-Low-Level-Vision")
*   [Awesome-NeurIPS2021-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-NeurIPS2021-Low-Level-Vision "Awesome-NeurIPS2021-Low-Level-Vision")
*   [Awesome-ICCV2021-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Kobaayyy/Awesome-ICCV2021-Low-Level-Vision "Awesome-ICCV2021-Low-Level-Vision")
*   [Awesome-CVPR2021/CVPR2020-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Kobaayyy/Awesome-CVPR2021-CVPR2020-Low-Level-Vision "Awesome-CVPR2021/CVPR2020-Low-Level-Vision")
*   [Awesome-ECCV2020-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Kobaayyy/Awesome-ECCV2020-Low-Level-Vision "Awesome-ECCV2020-Low-Level-Vision")
*   [IQA-PyTorch: A comprehensive toolbox for image quality assessment](https://github.com/chaofengc/IQA-PyTorch)
*   [BasicSR: Open Source Image and Video Restoration Toolbox](https://github.com/XPixelGroup/BasicSR)

*   https://aitechtogether.com/python/107134.html







