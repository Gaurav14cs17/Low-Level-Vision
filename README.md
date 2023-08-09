Low-level and High-level tasks
==============================

Low-level tasks: Common ones include Super-Resolution, denoise, deblur, dehze, low-light enhancement, deartifacts, etc. To put it simply, it is to restore a specific degraded image into a good-looking image. Now basically use the end-to-end model to learn the solution process of this kind of ill-posed problem. The objective indicators are mainly PSNR, SSIM, everyone The indicators are all set very high. Currently facing the following problems:

*   The generalization is poor. If you change the data set, the performance of the same task will be poor.
*   The existence of objective indicators and subjective feelings, GAP.
*   For the problem of landing, the SOTA model has a lot of computation (hundreds of G Flops), but it is actually impossible to use it like this.
*   It tends to solve practical problems, mainly serving people, such as various night scene modes and beautification in mobile phones, which will use related algorithms.
*   Low-level companies on the market are mostly mobile phone manufacturers (Huami OV), security ( [Hikang](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.zhihu.com/search?q%3D%25E6%25B5%25B7%25E5%25BA%25B7%25E5%25A4%25A7%25E5%258D%258E%26search_source%3DEntity%26hybrid_search_source%3DEntity%26hybrid_search_extra%3D%257B%2522sourceType%2522%253A%2522answer%2522%252C%2522sourceId%2522%253A2251241448%257D "Hikvision Dahua") Dahua), cameras (DJI, ISP manufacturers), drones (DJI), video websites (Bilibili, Kuaishou, etc.) ). Generally, scenes involving image and video enhancement are low-level trial problems.

  
High-level tasks: classification, detection, segmentation, etc. Generally, the public training data are high-quality images. When sending degraded images, the performance will decrease, even if the network has undergone a large amount of data enhancement (shape, brightness, chroma, etc. transformation). It is impossible for real application scenarios to be as perfect as the training set. There will be various degradation problems in the process of collecting images, and a combination of the two is required. In simple terms, the combination methods are divided into the following

*   Fine-tuning directly on the degraded image
*   First go through the low-level enhanced network, and then send it to the high-level model, and the two are trained separately
*   Joint training of augmented network and high-level models (such as classification)

**Table of contents**

* * *

CVPR2023-Low-Level-Vision
=========================

Image Restoration – Image Restoration
-------------------------------------

**Efficient and Explicit Modeling of Image Hierarchies for Image Restoration**

*   Paper:  [https://arxiv.org/abs/2303.00748](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.00748 "https://arxiv.org/abs/2303.00748")
*   Code:  [GitHub – ofsoundof/GRL-Image-Restoration](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ofsoundof/GRL-Image-Restoration "GitHub - ofsoundof/GRL-Image-Restoration")
*   Tags: Transformer

**Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**

*   Paper:  [https://arxiv.org/abs/2303.06859](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.06859 "https://arxiv.org/abs/2303.06859")
*   Code:  [https://github.com/lixinustc/Casual-IRDIL](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lixinustc/Casual-IRDIL "https://github.com/lixinustc/Casual-IRDIL")

**Generative Diffusion Prior for Unified Image Restoration and Enhancement**

*   Paper:  [https://arxiv.org/abs/2304.01247](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.01247 "https://arxiv.org/abs/2304.01247")
*   Code:  [https://github.com/Fayeben/GenerativeDiffusionPrior](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Fayeben/GenerativeDiffusionPrior "https://github.com/Fayeben/GenerativeDiffusionPrior")

**Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank**

*   Paper:  [https://arxiv.org/abs/2303.09101](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.09101 "https://arxiv.org/abs/2303.09101")
*   Code:  [https://github.com/Huang-ShiRui/Semi-UIR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Huang-ShiRui/Semi-UIR "https://github.com/Huang-ShiRui/Semi-UIR")
*   Tags: Underwater Image Restoration

**Nighttime Smartphone Reflective Flare Removal Using Optical Center Symmetry Prior**

*   Paper:  [https://arxiv.org/abs/2303.15046](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.15046 "https://arxiv.org/abs/2303.15046")
*   Code:  [https://github.com/ykdai/BracketFlare](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ykdai/BracketFlare "https://github.com/ykdai/BracketFlare")
*   Tags: Reflective Flare Removal

### Image Reconstruction

**Raw Image Reconstruction with Learned Compact Metadata**

*   Paper:  [https://arxiv.org/abs/2302.12995](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2302.12995 "https://arxiv.org/abs/2302.12995")
*   Code:  [GitHub – wyf0912/R2LCM: \[CVPR 2023\] Raw Image Reconstruction with Learned Compact Metadata](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wyf0912/R2LCM "GitHub - wyf0912/R2LCM: [CVPR 2023] Raw Image Reconstruction with Learned Compact Metadata")

**High-resolution image reconstruction with latent diffusion models from human brain activity**

*   Paper:  [High-resolution image reconstruction with latent diffusion models from human brain activity | bioRxiv](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.biorxiv.org/content/10.1101/2022.11.18.517004v2 "High-resolution image reconstruction with latent diffusion models from human brain activity | bioRxiv")
*   Code:  [GitHub – yu-takagi/StableDiffusionReconstruction: Takagi and Nishimoto, CVPR 2023](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yu-takagi/StableDiffusionReconstruction "GitHub - yu-takagi/StableDiffusionReconstruction: Takagi and Nishimoto, CVPR 2023")

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**

*   Paper:  [https://arxiv.org/abs/2303.06885](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.06885 "https://arxiv.org/abs/2303.06885")

### Burst Restoration

**Burstormer: Burst Image Restoration and Enhancement Transformer**

*   Paper:  [https://arxiv.org/abs/2304.01194](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.01194 "https://arxiv.org/abs/2304.01194")
*   Code:  [GitHub – akshaydudhane16/Burstormer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/akshaydudhane16/Burstormer "GitHub - akshaydudhane16/Burstormer")

### Video Restoration

**Blind Video Deflickering by Neural Filtering with a Flawed Atlas**

*   Paper:  [https://arxiv.org/abs/2303.08120](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.08120 "https://arxiv.org/abs/2303.08120")
*   Code:  [GitHub – ChenyangLEI/All-In-One-Deflicker: \[CVPR2023\] Blind Video Deflickering by Neural Filtering with a Flawed Atlas](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ChenyangLEI/All-In-One-Deflicker "GitHub - ChenyangLEI/All-In-One-Deflicker: [CVPR2023] Blind Video Deflickering by Neural Filtering with a Flawed Atlas")
*   Tags: Deflickering

Super Resolution – super resolution
-----------------------------------

### Image Super Resolution

**Activating More Pixels in Image Super-Resolution Transformer**

*   Paper:  [https://arxiv.org/abs/2205.04437](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.04437 "https://arxiv.org/abs/2205.04437")
*   Code:  [https://github.com/XPixelGroup/HAT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/XPixelGroup/HAT "https://github.com/XPixelGroup/HAT")
*   Tags: Transformer

**N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2211.11436](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.11436 "https://arxiv.org/abs/2211.11436")
*   Code:  [https://github.com/rami0205/NGramSwin](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/rami0205/NGramSwin "https://github.com/rami0205/NGramSwin")

**Omni Aggregation Networks for Lightweight Image Super-Resolution**

*   Paper:
*   Code:  [GitHub – Francis0625/Omni-SR: \[CVPR2023\] Implementation of ”Omni Aggregation Networks for Lightweight Image Super-Resolution”.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Francis0625/Omni-SR "GitHub - Francis0625/Omni-SR: [CVPR2023] Implementation of ''Omni Aggregation Networks for Lightweight Image Super-Resolution")

**OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2303.01091](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.01091 "https://arxiv.org/abs/2303.01091")

**Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2303.05156](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.05156 "https://arxiv.org/abs/2303.05156")

**Super-Resolution Neural Operator**

*   Paper:  [https://arxiv.org/abs/2303.02584](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.02584 "https://arxiv.org/abs/2303.02584")
*   Code:  [https://github.com/2y7c3/Super-Resolution-Neural-Operator](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/2y7c3/Super-Resolution-Neural-Operator "https://github.com/2y7c3/Super-Resolution-Neural-Operator")

**Human Guided Ground-truth Generation for Realistic Image Super-resolution**

*   Paper:  [https://arxiv.org/abs/2303.13069](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.13069 "https://arxiv.org/abs/2303.13069")
*   Code:  [https://github.com/ChrisDud0257/PosNegGT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ChrisDud0257/PosNegGT "https://github.com/ChrisDud0257/PosNegGT")

**Implicit Diffusion Models for Continuous Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2303.16491](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.16491 "https://arxiv.org/abs/2303.16491")
*   Code:  [https://github.com/Ree1s/IDM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Ree1s/IDM "https://github.com/Ree1s/IDM")

**Zero-Shot Dual-Lens Super-Resolution**

*   Paper:
*   Code:  [https://github.com/XrKang/ZeDuSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/XrKang/ZeDuSR "https://github.com/XrKang/ZeDuSR")

**Learning Generative Structure Prior for Blind Text Image Super-resolution**

*   Paper:  [https://arxiv.org/abs/2303.14726](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14726 "https://arxiv.org/abs/2303.14726")
*   Code:  [https://github.com/csxmli2016/MARCONet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/csxmli2016/MARCONet "https://github.com/csxmli2016/MARCONet")
*   Tags: Text SR

**Guided Depth Super-Resolution by Deep Anisotropic Diffusion**

*   Paper:  [https://arxiv.org/abs/2211.11592](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.11592 "https://arxiv.org/abs/2211.11592")
*   Code:  [GitHub – prs-eth/Diffusion-Super-Resolution: \[CVPR 2023\] Guided Depth Super-Resolution by Deep Anisotropic Diffusion](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/prs-eth/Diffusion-Super-Resolution "GitHub - prs-eth/Diffusion-Super-Resolution: [CVPR 2023] Guided Depth Super-Resolution by Deep Anisotropic Diffusion")
*   Tags: Guided Depth SR

### Video Super Resolution

**Towards High-Quality and Efficient Video Super-Resolution via Spatial-Temporal Data Overfitting**

*   Paper:  [https://arxiv.org/abs/2303.08331](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.08331 "https://arxiv.org/abs/2303.08331")
*   Code:  [coulsonlee/STDO-CVPR2023 GitHub](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/coulsonlee/STDO-CVPR2023 "coulsonlee/STDO-CVPR2023 GitHub")

**Structured Sparsity Learning for Efficient Video Super-Resolution**

*   Paper:  [https://github.com/Zj-BinXia/SSL](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Zj-BinXia/SSL "https://github.com/Zj-BinXia/SSL")
*   Code:  [https://arxiv.org/abs/2206.07687](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.07687 "https://arxiv.org/abs/2206.07687")

Image Rescaling – image scaling
-------------------------------

**HyperThumbnail: Real-time 6K Image Rescaling with Rate-distortion Optimization**

*   Paper:  [https://arxiv.org/abs/2304.01064](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.01064 "https://arxiv.org/abs/2304.01064")
*   Code:  [GitHub – AbnerVictor/HyperThumbnail: \[CVPR 2023\] HyperThumbnail: Real-time 6K Image Rescaling with Rate-distortion Optimization. Official implementation.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/AbnerVictor/HyperThumbnail "GitHub - AbnerVictor/HyperThumbnail: [CVPR 2023] HyperThumbnail: Real-time 6K Image Rescaling with Rate-distortion Optimization. Official implementation.")

Denoising – denoising
---------------------

### Image Denoising

**Masked Image Training for Generalizable Deep Image Denoising**

*   Paper:  [https://arxiv.org/abs/2303.13132](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.13132 "https://arxiv.org/abs/2303.13132")
*   Code:  [https://github.com/haoyuc/MaskedDenoising](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/haoyuc/MaskedDenoising "https://github.com/haoyuc/MaskedDenoising")

**Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising**

*   Paper:  [https://arxiv.org/abs/2303.14934](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14934 "https://arxiv.org/abs/2303.14934")
*   Cdoe:  [https://github.com/nagejacob/SpatiallyAdaptiveSSID](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/nagejacob/SpatiallyAdaptiveSSID "https://github.com/nagejacob/SpatiallyAdaptiveSSID")
*   Tags: Self-Supervised

**LG-BPN: Local and Global Blind-Patch Network for Self-Supervised Real-World Denoising**

*   Paper:  [https://arxiv.org/abs/2304.00534](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.00534 "https://arxiv.org/abs/2304.00534")
*   Code:  [https://github.com/Wang-XIaoDingdd/LGBPN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Wang-XIaoDingdd/LGBPN "https://github.com/Wang-XIaoDingdd/LGBPN")
*   Tags: Self-Supervised

**Real-time Controllable Denoising for Image and Video**

*   Paper:  [https://arxiv.org/pdf/2303.16425.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/pdf/2303.16425.pdf "https://arxiv.org/pdf/2303.16425.pdf")

Deblurring – Deblurring
-----------------------

### Image Deblurring

**Structured Kernel Estimation for Photon-Limited Deconvolution**

*   Paper:  [https://arxiv.org/abs/2303.03472](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.03472 "https://arxiv.org/abs/2303.03472")
*   Code:  [https://github.com/sanghviyashiitb/structured-kernel-cvpr23](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sanghviyashiitb/structured-kernel-cvpr23 "https://github.com/sanghviyashiitb/structured-kernel-cvpr23")

**Blur Interpolation Transformer for Real-World Motion from Blur**

*   Paper:  [https://arxiv.org/abs/2211.11423](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.11423 "https://arxiv.org/abs/2211.11423")
*   Code:  [https://github.com/zzh-tech/BiT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zzh-tech/BiT "https://github.com/zzh-tech/BiT")

**Neumann Network with Recursive Kernels for Single Image Defocus Deblurring**

*   Paper:
*   Code:  [https://github.com/csZcWu/NRKNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/csZcWu/NRKNet "https://github.com/csZcWu/NRKNet")

**Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring**

*   Paper:  [https://arxiv.org/abs/2211.12250](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.12250 "https://arxiv.org/abs/2211.12250")
*   Code:  [GitHub – kkkls/FFTformer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/kkkls/FFTformer "GitHub - kkkls/FFTformer")

Deraining – deraining
---------------------

**Learning A Sparse Transformer Network for Effective Image Deraining**

*   Paper:  [https://arxiv.org/abs/2303.11950](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.11950 "https://arxiv.org/abs/2303.11950")
*   Code:  [https://github.com/cschenxiang/DRSformer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/cschenxiang/DRSformer "https://github.com/cschenxiang/DRSformer")

Dehazing – to fog
-----------------

**RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors**

*   Paper:
*   Code:  [GitHub – RQ-Wu/RIDCP\_dehazing: \[CVPR 2023\] | RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/RQ-Wu/RIDCP "GitHub - RQ-Wu/RIDCP_dehazing: [CVPR 2023] | RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors")

**Curricular Contrastive Regularization for Physics-aware Single Image Dehazing**

*   Paper:  [https://arxiv.org/abs/2303.14218](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14218 "https://arxiv.org/abs/2303.14218")
*   Code:  [GitHub – YuZheng9/C2PNet: \[CVPR 2023\] Curricular Contrastive Regularization for Physics-aware Single Image Dehazing](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/YuZheng9/C2PNet "GitHub - YuZheng9/C2PNet: [CVPR 2023] Curricular Contrastive Regularization for Physics-aware Single Image Dehazing")

**Video Dehazing via a Multi-Range Temporal Alignment Network with Physical Prior**

*   Paper:  [https://arxiv.org/abs/2303.09757](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.09757 "https://arxiv.org/abs/2303.09757")
*   Code:  [https://github.com/jiaqixuac/MAP-Net](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jiaqixuac/MAP-Net "https://github.com/jiaqixuac/MAP-Net")

HDR Imaging / Multi-Exposure Image Fusion – HDR image generation / multi-exposure image fusion
----------------------------------------------------------------------------------------------

**Learning a Practical SDR-to-HDRTV Up-conversion using New Dataset and Degradation Models**

*   Paper:  [https://arxiv.org/abs/2303.13031](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.13031 "https://arxiv.org/abs/2303.13031")
*   Code:  [https://github.com/AndreGuo/HDRTVDM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/AndreGuo/HDRTVDM "https://github.com/AndreGuo/HDRTVDM")

Frame Interpolation – frame insertion
-------------------------------------

**Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2303.00440](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.00440 "https://arxiv.org/abs/2303.00440")
*   Code:  [GitHub – MCG-NJU/EMA-VFI: \[CVPR 2023\] Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolatio](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/MCG-NJU/EMA-VFI "GitHub - MCG-NJU/EMA-VFI: [CVPR 2023] Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation")

**A Unified Pyramid Recurrent Network for Video Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2211.03456](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.03456 "https://arxiv.org/abs/2211.03456")
*   Code:  [GitHub – srcn-ivl/UPR-Net: Official implementation of our CVPR2023 paper “A Unified Pyramid Recurrent Network for Video Frame Interpolation”](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/srcn-ivl/UPR-Net "GitHub - srcn-ivl/UPR-Net: Official implementation of our CVPR2023 paper")

**BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2304.02225](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.02225 "https://arxiv.org/abs/2304.02225")
*   Code:  [GitHub – JunHeum/BiFormer: BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation, CVPR2023](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JunHeum/BiFormer "GitHub - JunHeum/BiFormer: BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation, CVPR2023")

**Event-based Video Frame Interpolation with Cross-Modal Asymmetric Bidirectional Motion Fields**

*   Paper:
*   Code:  [GitHub – intelpro/CBMNet: Official repository of “Event-based Video Frame Interpolation with Cross-Modal Asymmetric Bidirectional Motion Fields”, CVPR 2023 paper](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/intelpro/CBMNet "GitHub - intelpro/CBMNet: Official repository of")
*   Tags: Event-based

**Event-based Blurry Frame Interpolation under Blind Exposure**

*   Paper:
*   Code:  [GitHub – WarranWeng/EBFI-BE: Event-based Blurry Frame Interpolation under Blind Exposure, CVPR2023](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/WarranWeng/EBFI-BE "GitHub - WarranWeng/EBFI-BE: Event-based Blurry Frame Interpolation under Blind Exposure, CVPR2023")
*   Tags: Event-based

**Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time**

*   Paper:  [https://arxiv.org/abs/2303.15043](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.15043 "https://arxiv.org/abs/2303.15043")
*   Code:  [GitHub – shangwei5/VIDUE: Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time (CVPR2023)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/shangwei5/VIDUE "GitHub - shangwei5/VIDUE: Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time (CVPR2023)")
*   Tags: Frame Interpolation and Deblurring

Image Enhancement – ​​image enhancement
---------------------------------------

### Low-Light Image Enhancement

**Learning Semantic-Aware Knowledge Guidance for Low-Light Image Enhancement**

*   Paper:
*   Code:  [https://github.com/langmanbusi/Semantic-Aware-Low-Light-Image-Enhancement](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/langmanbusi/Semantic-Aware-Low-Light-Image-Enhancement "https://github.com/langmanbusi/Semantic-Aware-Low-Light-Image-Enhancement")

**Visibility Constrained Wide-band Illumination Spectrum Design for Seeing-in-the-Dark**

*   Paper:  [https://arxiv.org/abs/2303.11642](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.11642 "https://arxiv.org/abs/2303.11642")
*   Code:  [https://github.com/MyNiuuu/VCSD](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/MyNiuuu/VCSD "https://github.com/MyNiuuu/VCSD")
*   Tags: NIR2RGB

Image Matting – image matting
-----------------------------

**Referring Image Matting**

*   Paper:  [https://arxiv.org/abs/2206.05149](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.05149 "https://arxiv.org/abs/2206.05149")
*   Code:  [GitHub – JizhiziLi/RIM: \[CVPR 2023\] Referring Image Matting](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JizhiziLi/RIM "GitHub - JizhiziLi/RIM: [CVPR 2023] Referring Image Matting")

Shadow Removal – shadow removal
-------------------------------

**ShadowDiffusion: When Degradation Prior Meets Diffusion Model for Shadow Removal**

*   Paper:  [https://arxiv.org/abs/2212.04711](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2212.04711 "https://arxiv.org/abs/2212.04711")
*   Code:  [https://github.com/GuoLanqing/ShadowDiffusion](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/GuoLanqing/ShadowDiffusion "https://github.com/GuoLanqing/ShadowDiffusion")

Image Compression – image compression
-------------------------------------

**Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger**

*   Paper:  [https://arxiv.org/abs/2302.14677](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2302.14677 "https://arxiv.org/abs/2302.14677")

**Context-based Trit-Plane Coding for Progressive Image Compression**

*   Paper:  [https://arxiv.org/abs/2303.05715](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.05715 "https://arxiv.org/abs/2303.05715")
*   Code:  [https://github.com/seungminjeon-github/CTC](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/seungminjeon-github/CTC "https://github.com/seungminjeon-github/CTC")

**Learned Image Compression with Mixed Transformer-CNN Architectures**

*   Paper:  [https://arxiv.org/abs/2303.14978](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14978 "https://arxiv.org/abs/2303.14978")
*   Code:  [GitHub – jmliu206/LIC\_TCM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jmliu206/LIC_TCM "GitHub - jmliu206/LIC_TCM")

### Video Compression

**Neural Video Compression with Diverse Contexts**

*   Paper:  [https://github.com/microsoft/DCVC](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/microsoft/DCVC "https://github.com/microsoft/DCVC")
*   Code:  [https://arxiv.org/abs/2302.14402](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2302.14402 "https://arxiv.org/abs/2302.14402")

Image Quality Assessment – ​​image quality assessment
-----------------------------------------------------

**Quality-aware Pre-trained Models for Blind Image Quality Assessment**

*   Paper:  [https://arxiv.org/abs/2303.00521](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.00521 "https://arxiv.org/abs/2303.00521")

**Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective**

*   Paper:  [https://arxiv.org/abs/2303.14968](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14968 "https://arxiv.org/abs/2303.14968")
*   Code:  [GitHub – zwx8981/LIQE](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zwx8981/LIQE "GitHub - zwx8981/LIQE")

**Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method**

*   Paper:  [https://arxiv.org/abs/2303.15166](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.15166 "https://arxiv.org/abs/2303.15166")
*   Code:  [GitHub – Dreemurr-T/BAID](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Dreemurr-T/BAID "GitHub - Dreemurr-T/BAID")

**Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild**

*   Paper:  [https://arxiv.org/abs/2304.00451](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.00451 "https://arxiv.org/abs/2304.00451")

Style Transfer – style transfer
-------------------------------

**Fix the Noise: Disentangling Source Feature for Controllable Domain Translation**

*   Paper:  [https://arxiv.org/abs/2303.11545](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.11545 "https://arxiv.org/abs/2303.11545")
*   Code:  [https://github.com/LeeDongYeun/FixNoise](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/LeeDongYeun/FixNoise "https://github.com/LeeDongYeun/FixNoise")

**Neural Preset for Color Style Transfer**

*   Paper:  [https://arxiv.org/abs/2303.13511](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.13511 "https://arxiv.org/abs/2303.13511")
*   Code:  [https://github.com/ZHKKKe/NeuralPreset](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ZHKKKe/NeuralPreset "https://github.com/ZHKKKe/NeuralPreset")

**CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer**

*   Paper:  [https://arxiv.org/abs/2303.17867](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.17867 "https://arxiv.org/abs/2303.17867")

**StyleGAN Salon: Multi-View Latent Optimization for Pose-Invariant Hairstyle Transfer**

*   Paper:  [https://arxiv.org/abs/2304.02744](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.02744 "https://arxiv.org/abs/2304.02744")
*   Project:  [StyleGANSalon](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://stylegan-salon.github.io/ "StyleGANSalon")

Image Editing – image editing
-----------------------------

**Imagic: Text-Based Real Image Editing with Diffusion Models**

*   Paper:  [https://arxiv.org/abs/2210.09276](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2210.09276 "https://arxiv.org/abs/2210.09276")

**SINE: SINgle Image Editing with Text-to-Image Diffusion Models**

*   Paper:  [https://arxiv.org/abs/2212.04489](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2212.04489 "https://arxiv.org/abs/2212.04489")
*   Code:  [https://github.com/zhang-zx/SINE](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zhang-zx/SINE "https://github.com/zhang-zx/SINE")

**CoralStyleCLIP: Co-optimized Region and Layer Selection for Image Editing**

*   Paper:  [https://arxiv.org/abs/2303.05031](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.05031 "https://arxiv.org/abs/2303.05031")

**DeltaEdit: Exploring Text-free Training for Text-Driven Image Manipulation**

*   Paper:  [https://arxiv.org/abs/2303.06285](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.06285 "https://arxiv.org/abs/2303.06285")
*   Code:  [https://arxiv.org/abs/2303.06285](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.06285 "https://arxiv.org/abs/2303.06285")

**SIEDOB: Semantic Image Editing by Disentangling Object and Background**

*   Paper:  [https://arxiv.org/abs/2303.13062](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.13062 "https://arxiv.org/abs/2303.13062")
*   Code:  [GitHub – WuyangLuo/SIEDOB](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/WuyangLuo/SIEDOB "GitHub - WuyangLuo/SIEDOB")

Image Generation/Synthesis / Image-to-Image Translation – Image Generation/Synthesis/Translation
------------------------------------------------------------------------------------------------

### Text-to-Image / Text Guided / Multi-Modal

**Multi-Concept Customization of Text-to-Image Diffusion**

*   Paper:  [https://arxiv.org/abs/2212.04488](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2212.04488 "https://arxiv.org/abs/2212.04488")
*   Code:  [GitHub – adobe-research/custom-diffusion: Custom Diffusion: Multi-Concept Customization of Text-to-Image Diffusion (CVPR 2023)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/adobe-research/custom-diffusion "GitHub - adobe-research/custom-diffusion: Custom Diffusion: Multi-Concept Customization of Text-to-Image Diffusion (CVPR 2023)")

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2301.12959](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2301.12959 "https://arxiv.org/abs/2301.12959")
*   Code:  [GitHub – tobran/GALIP: \[CVPR2023\] A faster, smaller, and better text-to-image model for large-scale training](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/tobran/GALIP "GitHub - tobran/GALIP: [CVPR2023] A faster, smaller, and better text-to-image model for large-scale training")

**Scaling up GANs for Text-to-Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2303.05511](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.05511 "https://arxiv.org/abs/2303.05511")
*   Project:  [GigaGAN: Scaling up GANs for Text-to-Image Synthesis](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://mingukkang.github.io/GigaGAN/ "GigaGAN: Scaling up GANs for Text-to-Image Synthesis")

**MAGVLT: Masked Generative Vision-and-Language Transformer**

*   Paper:  [https://arxiv.org/abs/2303.12208](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.12208 "https://arxiv.org/abs/2303.12208")

**Freestyle Layout-to-Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2303.14412](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14412 "https://arxiv.org/abs/2303.14412")
*   Code:  [GitHub – essunny310/FreestyleNet: \[CVPR 2023 Highlight\] Freestyle Layout-to-Image Synthesis](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/essunny310/FreestyleNet "GitHub - essunny310/FreestyleNet: [CVPR 2023 Highlight] Freestyle Layout-to-Image Synthesis")

**Variational Distribution Learning for Unsupervised Text-to-Image Generation**

*   Paper:  [https://arxiv.org/abs/2303.16105](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.16105 "https://arxiv.org/abs/2303.16105")

**Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment**

*   Paper:  [https://arxiv.org/abs/2303.17490](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.17490 "https://arxiv.org/abs/2303.17490")
*   Project:  [Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://sound2scene.github.io/ "Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment")

**Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation**

*   Paper:  [https://arxiv.org/abs/2304.01816](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.01816 "https://arxiv.org/abs/2304.01816")

### Image-to-Image / Image Guided

**LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data**

*   Paper:  [https://arxiv.org/abs/2208.14889](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.14889 "https://arxiv.org/abs/2208.14889")
*   Code:  [GitHub – KU-CVLAB/LANIT: Official repository for LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data (CVPR 2023)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/KU-CVLAB/LANIT "GitHub - KU-CVLAB/LANIT: Official repository for LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data (CVPR 2023)")

**Person Image Synthesis via Denoising Diffusion Model**

*   Paper:  [https://arxiv.org/abs/2211.12500](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.12500 "https://arxiv.org/abs/2211.12500")
*   Code:  [https://github.com/ankanbhunia/PIDM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ankanbhunia/PIDM "https://github.com/ankanbhunia/PIDM")

**Picture that Sketch: Photorealistic Image Generation from Abstract Sketches**

*   Paper:  [https://arxiv.org/abs/2303.11162](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.11162 "https://arxiv.org/abs/2303.11162")

**Fine-Grained Face Swapping via Regional GAN ​​Inversion**

*   Paper:  [https://arxiv.org/abs/2211.14068](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.14068 "https://arxiv.org/abs/2211.14068")
*   Code:  [https://github.com/e4s2022/e4s](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/e4s2022/e4s "https://github.com/e4s2022/e4s")

**Masked and Adaptive Transformer for Exemplar Based Image Translation**

*   Paper:  [https://arxiv.org/abs/2303.17123](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.17123 "https://arxiv.org/abs/2303.17123")
*   Code:  [GitHub – AiArt-HDU/MATEBIT: Source code of “Masked and Adaptive Transformer for Exemplar Based Image Translation”, accepted by CVPR 2023.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/AiArt-HDU/MATEBIT "GitHub - AiArt-HDU/MATEBIT: Source code of")

**Zero-shot Generative Model Adaptation via Image-specific Prompt Learning**

*   Paper:  [https://arxiv.org/abs/2304.03119](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.03119 "https://arxiv.org/abs/2304.03119")
*   Code:  [GitHub – Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation: \[CVPR 2023\] Zero-shot Generative Model Adaptation via Image-specific Prompt Learning](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation "GitHub - Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation: [CVPR 2023] Zero-shot Generative Model Adaptation via Image-specific Prompt Learning")

### Others for image generation

**AdaptiveMix: Robust Feature Representation via Shrinking Feature Space**

*   Paper:  [https://arxiv.org/abs/2303.01559](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.01559 "https://arxiv.org/abs/2303.01559")
*   Code:  [GitHub – WentianZhang-ML/AdaptiveMix: This is an official pytorch implementation of 'AdaptiveMix: Robust Feature Representation via Shrinking Feature Space' (accepted by CVPR2023).](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/WentianZhang-ML/AdaptiveMix "GitHub - WentianZhang-ML/AdaptiveMix: This is an official pytorch implementation of 'AdaptiveMix: Robust Feature Representation via Shrinking Feature Space' (accepted by CVPR2023).")

**MAGE: MASKed Generative Encoder to Unify Representation Learning and Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2211.09117](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.09117 "https://arxiv.org/abs/2211.09117")
*   Code:  [GitHub – LTH14/mage: A PyTorch implementation of MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/LTH14/mage "GitHub - LTH14/mage: A PyTorch implementation of MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis")

**Regularized Vector Quantization for Tokenized Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2303.06424](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.06424 "https://arxiv.org/abs/2303.06424")

**Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization**

*   Paper:
*   Code:  [https://github.com/CrossmodalGroup/DynamicVectorQuantization](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/CrossmodalGroup/DynamicVectorQuantization "https://github.com/CrossmodalGroup/DynamicVectorQuantization")

**Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation**

*   Paper:
*   Code:  [https://github.com/CrossmodalGroup/MaskedVectorQuantization](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/CrossmodalGroup/MaskedVectorQuantization "https://github.com/CrossmodalGroup/MaskedVectorQuantization")

**Exploring Incompatible Knowledge Transfer in Few-shot Image Generation**

*   Paper:
*   Code:  [GitHub – yunqing-me/RICK: The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yunqing-me/RICK "GitHub - yunqing-me/RICK: The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023")

**Post-training Quantization on Diffusion Models**

*   Paper:  [https://arxiv.org/abs/2211.15736](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.15736 "https://arxiv.org/abs/2211.15736")
*   Code:  [GitHub – 42Shawn/PTQ4DM: Implementation of Post-training Quantization on Diffusion Models (CVPR 2023)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/42Shawn/PTQ4DM "GitHub - 42Shawn/PTQ4DM: Implementation of Post-training Quantization on Diffusion Models (CVPR 2023)")

**LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation**

*   Paper:  [https://arxiv.org/abs/2303.17189](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.17189 "https://arxiv.org/abs/2303.17189")
*   Code:  [GitHub – ZGCTroy/LayoutDiffusion](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ZGCTroy/LayoutDiffusion "GitHub - ZGCTroy/LayoutDiffusion")

**DiffCollage: Parallel Generation of Large Content with Diffusion Models**

*   Paper:  [https://arxiv.org/abs/2303.17076](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.17076 "https://arxiv.org/abs/2303.17076")
*   Project:  [DiffCollage: Parallel Generation of Large Content with Diffusion Models](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://research.nvidia.com/labs/dir/diffcollage/ "DiffCollage: Parallel Generation of Large Content with Diffusion Models")

**Few-shot Semantic Image Synthesis with Class Affinity Transfer**

*   Paper:  [https://arxiv.org/abs/2304.02321](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.02321 "https://arxiv.org/abs/2304.02321")

### Video Generation

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**

*   Paper:  [https://arxiv.org/abs/2303.13744](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.13744 "https://arxiv.org/abs/2303.13744")
*   Code:  [GitHub – nihaomiao/CVPR23\_LFDM: The pytorch implementation of our CVPR 2023 paper “Conditional Image-to-Video Generation with Latent Flow Diffusion Models”](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/nihaomiao/CVPR2023_LFDM "GitHub - nihaomiao/CVPR23_LFDM: The pytorch implementation of our CVPR 2023 paper")

**Video Probabilistic Diffusion Models in Projected Latent Space**

*   Paper:  [https://arxiv.org/abs/2302.07685](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2302.07685 "https://arxiv.org/abs/2302.07685")
*   Code:  [https://github.com/sihyun-yu/PVDM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sihyun-yu/PVDM "https://github.com/sihyun-yu/PVDM")

**DPE: Disentanglement of Pose and Expression for General Video Portrait Editing**

*   Paper:  [https://arxiv.org/abs/2301.06281](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2301.06281 "https://arxiv.org/abs/2301.06281")
*   Code:  [GitHub – Carlyx/DPE: \[CVPR 2023\] DPE: Disentanglement of Pose and Expression for General Video Portrait Editing](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Carlyx/DPE "GitHub - Carlyx/DPE: [CVPR 2023] DPE: Disentanglement of Pose and Expression for General Video Portrait Editing")

**Decomposed Diffusion Models for High-Quality Video Generation**

*   Paper:  [https://arxiv.org/abs/2303.08320](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.08320 "https://arxiv.org/abs/2303.08320")

**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding**

*   Paper:  [https://arxiv.org/abs/2212.02802](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2212.02802 "https://arxiv.org/abs/2212.02802")
*   Code:  [GitHub – man805/Diffusion-Video-Autoencoders: An official implementation of “Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding” (CVPR 2023) in PyTorch.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/man805/Diffusion-Video-Autoencoders "GitHub - man805/Diffusion-Video-Autoencoders: An official implementation of")

**MoStGAN: Video Generation with Temporal Motion Styles**

*   Paper:
*   Code:  [https://github.com/xiaoqian-shen/MoStGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/xiaoqian-shen/MoStGAN "https://github.com/xiaoqian-shen/MoStGAN")

### Others

**DC2: Dual-Camera Defocus Control by Learning to Refocus**

*   Paper:  [https://arxiv.org/abs/2304.03285](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.03285 "https://arxiv.org/abs/2304.03285")
*   Project:  [DC2: Dual-Camera Defocus Control by Learning to Refocus](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://defocus-control.github.io/ "DC2: Dual-Camera Defocus Control by Learning to Refocus")

**Images Speak in Images: A Generalist Painter for In-Context Visual Learning**

*   Paper:  [https://arxiv.org/abs/2212.02499](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2212.02499 "https://arxiv.org/abs/2212.02499")
*   Code:  [https://github.com/baaivision/Painter](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/baaivision/Painter "https://github.com/baaivision/Painter")

**Unifying Layout Generation with a Decoupled Diffusion Model**

*   Paper:  [https://arxiv.org/abs/2303.05049](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.05049 "https://arxiv.org/abs/2303.05049")

**Unsupervised Domain Adaptation with Pixel-level Discriminator for Image-aware Layout Generation**

*   Paper:  [https://arxiv.org/abs/2303.14377](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14377 "https://arxiv.org/abs/2303.14377")

**PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout**

*   Paper:  [https://arxiv.org/abs/2303.15937](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.15937 "https://arxiv.org/abs/2303.15937")
*   Code:  [https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023 "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023")

**LayoutDM: Discrete Diffusion Model for Controllable Layout Generation**

*   Paper:  [https://arxiv.org/abs/2303.08137](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.08137 "https://arxiv.org/abs/2303.08137")
*   Code:  [https://github.com/CyberAgentAILab/layout-dm](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/CyberAgentAILab/layout-dm "https://github.com/CyberAgentAILab/layout-dm")

**Make-A-Story: Visual Memory Conditioned Consistent Story Generation**

*   Paper:  [https://arxiv.org/abs/2211.13319](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2211.13319 "https://arxiv.org/abs/2211.13319")
*   Code:  [https://github.com/ubc-vision/Make-A-Story](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ubc-vision/Make-A-Story "https://github.com/ubc-vision/Make-A-Story")

**Cross-GAN Auditing: Unsupervised Identification of Attribute Level Similarities and Differences between Pretrained Generative Models**

*   Paper:  [https://arxiv.org/abs/2303.10774](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.10774 "https://arxiv.org/abs/2303.10774")
*   Code:  [mattolson93/cross\_gan\_auditing GitHub](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/mattolson93/cross_gan_auditing "mattolson93/cross_gan_auditing GitHub")

**LightPainter: Interactive Portrait Relighting with Freehand Scribble**

*   Paper:  [https://arxiv.org/abs/2303.12950](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.12950 "https://arxiv.org/abs/2303.12950")
*   Tags: Portrait Relighting

**Neural Texture Synthesis with Guided Correspondence**

*   Paper:
*   Code:  [https://github.com/EliotChenKJ/Guided-Correspondence-Loss](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/EliotChenKJ/Guided-Correspondence-Loss "https://github.com/EliotChenKJ/Guided-Correspondence-Loss")
*   Tags: Texture Synthesis

**CF-Font: Content Fusion for Few-shot Font Generation**

*   Paper:  [https://arxiv.org/abs/2303.14017](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14017 "https://arxiv.org/abs/2303.14017")
*   Code:  [https://github.com/wangchi95/CF-Font](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wangchi95/CF-Font "https://github.com/wangchi95/CF-Font")
*   Tags: Font Generation

**DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality**

*   Paper:  [https://arxiv.org/abs/2303.14585](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14585 "https://arxiv.org/abs/2303.14585")
*   Code:  [GitHub – yizhiwang96/deepvecfont-v2: \[CVPR 2023\] DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yizhiwang96/deepvecfont-v2 "GitHub - yizhiwang96/deepvecfont-v2: [CVPR 2023] DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality")

**Handwritten Text Generation from Visual Archetypes**

*   Paper:  [https://arxiv.org/abs/2303.15269](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.15269 "https://arxiv.org/abs/2303.15269")
*   Tags: Handwriting Generation

**Disentangling Writer and Character Styles for Handwriting Generation**

*   Paper:  [https://arxiv.org/abs/2303.14736](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.14736 "https://arxiv.org/abs/2303.14736")
*   Code:  [GitHub – dailenson/SDT: This repository is the official implementation of Disentangling Writer and Character Styles for Handwriting Generation (CVPR23).](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/dailenson/SDT "GitHub - dailenson/SDT: This repository is the official implementation of Disentangling Writer and Character Styles for Handwriting Generation (CVPR23).")
*   Tags: Handwriting Generation

**Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert**

*   Paper:  [https://arxiv.org/abs/2303.17480](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2303.17480 "https://arxiv.org/abs/2303.17480")
*   Code:  [GitHub – Sxjdwang/TalkLip](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Sxjdwang/TalkLip "GitHub - Sxjdwang/TalkLip")

**Uncurated Image-Text Datasets: Shedding Light on Demographic Bias**

*   Paper:  [https://arxiv.org/abs/2304.02828](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2304.02828 "https://arxiv.org/abs/2304.02828")
*   Code:  [https://github.com/noagarcia/phase](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/noagarcia/phase "https://github.com/noagarcia/phase")

CVPR2022-Low-Level-Vision
=========================

Image Restoration – Image Restoration
-------------------------------------

**Restorer: Efficient Transformer for High-Resolution Image Restoration**

*   Paper:  [https://arxiv.org/abs/2111.09881](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.09881 "https://arxiv.org/abs/2111.09881")
*   Code:  [https://github.com/swz30/Restormer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/swz30/Restormer "https://github.com/swz30/Restormer")
*   Tags: Transformer

**Uformer: A General U-Shaped Transformer for Image Restoration**

*   Paper:  [https://arxiv.org/abs/2106.03106](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2106.03106 "https://arxiv.org/abs/2106.03106")
*   Code:  [https://github.com/ZhendongWang6/Uformer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ZhendongWang6/Uformer "https://github.com/ZhendongWang6/Uformer")
*   Tags: Transformer

**MAXIM: Multi-Axis MLP for Image Processing**

*   Paper:  [https://arxiv.org/abs/2201.02973](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2201.02973 "https://arxiv.org/abs/2201.02973")
*   Code:  [https://github.com/google-research/maxim](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/google-research/maxim "https://github.com/google-research/maxim")
*   Tags: MLP, also do image enhancement

**All-In-One Image Restoration for Unknown Corruption**

*   Paper:  [http://pengxi.me/wp-content/uploads/2022/03/All-In-One-Image-Restoration-for-Unknown-Corruption.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=http://pengxi.me/wp-content/uploads/2022/03/All-In-One-Image-Restoration-for-Unknown-Corruption.pdf "http://pengxi.me/wp-content/uploads/2022/03/All-In-One-Image-Restoration-for-Unknown-Corruption.pdf")
*   Code:  [https://github.com/XLearning-SCU/2022-CVPR-AirNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/XLearning-SCU/2022-CVPR-AirNet "https://github.com/XLearning-SCU/2022-CVPR-AirNet")

**Fourier Document Restoration for Robust Document Dewarping and Recognition**

*   Paper:  [https://arxiv.org/abs/2203.09910](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.09910 "https://arxiv.org/abs/2203.09910")
*   Tags: Document Restoration

**Exploring and Evaluating Image Restoration Potential in Dynamic Scenes**

*   Paper:  [https://arxiv.org/abs/2203.11754](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.11754 "https://arxiv.org/abs/2203.11754")

**ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior**

*   Paper:  [https://arxiv.org/abs/2111.15362v2](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.15362v2 "https://arxiv.org/abs/2111.15362v2")
*   Code:  [https://github.com/ozgurkara99/ISNAS-DIP](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ozgurkara99/ISNAS-DIP "https://github.com/ozgurkara99/ISNAS-DIP")
*   Tags: DIP, NAS

**Deep Generalized Unfolding Networks for Image Restoration**

*   Paper:  [https://arxiv.org/abs/2204.13348](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.13348 "https://arxiv.org/abs/2204.13348")
*   Code:  [https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration "https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration")

**Attentive Fine-Grained Structured Sparsity for Image Restoration**

*   Paper:  [https://arxiv.org/abs/2204.12266](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.12266 "https://arxiv.org/abs/2204.12266")
*   Code:  [https://github.com/JungHunOh/SLS\_CVPR2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JungHunOh/SLS_CVPR2022 "https://github.com/JungHunOh/SLS_CVPR2022")

**Self-Supervised Deep Image Restoration via Adaptive Stochastic Gradient Langevin Dynamics**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Self-Supervised_Deep_Image_Restoration_via_Adaptive_Stochastic_Gradient_Langevin_Dynamics_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Self-Supervised

**KNN Local Attention for Image Restoration**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Lee_KNN_Local_Attention_for_Image_Restoration_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Code:  [https://sites.google.com/view/cvpr22-kit](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://sites.google.com/view/cvpr22-kit "https://sites.google.com/view/cvpr22-kit")

**GIQE: Generic Image Quality Enhancement via Nth Order Iterative Degradation**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Shyam_GIQE_Generic_Image_Quality_Enhancement_via_Nth_Order_Iterative_Degradation_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions**

*   Paper:  [https://arxiv.org/abs/2111.14813](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.14813 "https://arxiv.org/abs/2111.14813")
*   Code:  [https://github.com/jeya-maria-jose/TransWeather](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jeya-maria-jose/TransWeather "https://github.com/jeya-maria-jose/TransWeather")
*   Tags: Adverse Weather

**Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model**

*   Paper:  [https://openaccess.thecvf.com/content/CVPR2022/papers/Chen\_Learning\_Multiple\_Adverse\_Weather\_Removal\_via\_Two-Stage\_Knowledge\_Learning\_and\_CVPR\_2022\_paper.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.pdf "https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.pdf")
*   Code:  [https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal "https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal")
*   Tags: Adverse Weather(deraining, desnowing, dehazing)

**Rethinking Deep Face Restoration**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Rethinking_Deep_Face_Restoration_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: face

**RestoreFormer: High-Quality Blind Face Restoration From Ungraded Key-Value Pairs**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Wang_RestoreFormer_High-Quality_Blind_Face_Restoration_From_Undegraded_Key-Value_Pairs_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Code:  [https://github.com/wzhouxiff/RestoreFormer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wzhouxiff/RestoreFormer "https://github.com/wzhouxiff/RestoreFormer")
*   Tags: face

**Blind Face Restoration via Integrating Face Shape and Generative Priors**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Blind_Face_Restoration_via_Integrating_Face_Shape_and_Generative_Priors_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: face

**End-to-End Rubbing Restoration Using Generative Adversarial Networks**

*   Paper:  [https://arxiv.org/abs/2205.03743](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.03743 "https://arxiv.org/abs/2205.03743")
*   Code:  [https://github.com/qingfengtommy/RubbingGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/qingfengtommy/RubbingGAN "https://github.com/qingfengtommy/RubbingGAN")
*   Tags: \[Workshop\], Rubbing Restoration

**GenISP: Neural ISP for Low-Light Machine Cognition**

*   Paper:  [https://arxiv.org/abs/2205.03688](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.03688 "https://arxiv.org/abs/2205.03688")
*   Tags: \[Workshop\], ISP

### Burst Restoration

**A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift**

*   Paper:  [https://arxiv.org/abs/2203.09294](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.09294 "https://arxiv.org/abs/2203.09294")
*   Code:  [GitHub – GuoShi28/2StageAlign: The official codes of our CVPR2022 paper: A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/GuoShi28/2StageAlign "GitHub - GuoShi28/2StageAlign: The official codes of our CVPR2022 paper: A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift")
*   Tags: joint denoising and demosaicking

**Burst Image Restoration and Enhancement**

*   Paper:  [https://arxiv.org/abs/2110.03680](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2110.03680 "https://arxiv.org/abs/2110.03680")
*   Code:  [https://github.com/akshaydudhane16/BIPNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/akshaydudhane16/BIPNet "https://github.com/akshaydudhane16/BIPNet")

### Video Restoration

**Revisiting Temporal Alignment for Video Restoration**

*   Paper:  [https://arxiv.org/abs/2111.15288](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.15288 "https://arxiv.org/abs/2111.15288")
*   Code:  [GitHub – redrock303/Revisiting-Temporal-Alignment-for-Video-Restoration](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/redrock303/Revisiting-Temporal-Alignment-for-Video-Restoration "GitHub - redrock303/Revisiting-Temporal-Alignment-for-Video-Restoration")

**Neural Compression-Based Feature Learning for Video Restoration**

*   Paper: [https://arxiv.org/abs/2203.09208](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.09208 "https://arxiv.org/abs/2203.09208")

**Bringing Old Films Back to Life**

*   Paper:  [https://arxiv.org/abs/2203.17276](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.17276 "https://arxiv.org/abs/2203.17276")
*   Code:  [https://github.com/raywzy/Bringing-Old-Films-Back-to-Life](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/raywzy/Bringing-Old-Films-Back-to-Life "https://github.com/raywzy/Bringing-Old-Films-Back-to-Life")

**Neural Global Shutter: Learn to Restore Video from a Rolling Shutter Camera with Global Reset Feature**

*   Paper:  [https://arxiv.org/abs/2204.00974](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.00974 "https://arxiv.org/abs/2204.00974")
*   Code:  [https://github.com/lightChaserX/neural-global-shutter](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lightChaserX/neural-global-shutter "https://github.com/lightChaserX/neural-global-shutter")
*   Tags: restore clean global shutter (GS) videos

**Context-Aware Video Reconstruction for Rolling Shutter Cameras**

*   Paper:  [https://arxiv.org/abs/2205.12912](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.12912 "https://arxiv.org/abs/2205.12912")
*   Code:  [https://github.com/GitCVfb/CVR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/GitCVfb/CVR "https://github.com/GitCVfb/CVR")
*   Tags: Rolling Shutter Cameras

**E2V-SDE: From Asynchronous Events to Fast and Continuous Video Reconstruction via Neural Stochastic Differential Equations**

*   Paper:  [https://arxiv.org/abs/2206.07578](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.07578 "https://arxiv.org/abs/2206.07578")
*   Tags: Event camera
*   **Withdrawal due to plagiarism**

### Hyperspectral Image Reconstruction

**Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction**

*   Paper:  [https://arxiv.org/abs/2111.07910](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.07910 "https://arxiv.org/abs/2111.07910")
*   Code:  [https://github.com/caiyuanhao1998/MST](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/caiyuanhao1998/MST "https://github.com/caiyuanhao1998/MST")

**HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging**

*   Paper:  [https://arxiv.org/abs/2203.02149](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.02149 "https://arxiv.org/abs/2203.02149")

Super Resolution – super resolution
-----------------------------------

### Image Super Resolution

**Reflash Dropout in Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2112.12089](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.12089 "https://arxiv.org/abs/2112.12089")
*   Code:  [https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution "https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution")

**Residual Local Feature Network for Efficient Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2205.07514](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.07514 "https://arxiv.org/abs/2205.07514")
*   Code:  [https://github.com/fyan111/RLFN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fyan111/RLFN "https://github.com/fyan111/RLFN")
*   Tags: won the first place in the runtime track of the NTIRE 2022 efficient super-resolution challenge

**Learning the Degradation Distribution for Blind Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2203.04962](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.04962 "https://arxiv.org/abs/2203.04962")
*   Code:  [GitHub – greatlog/UnpairedSR: This is an offical implementation of the CVPR2022's paper \[Learning the Degradation Distribution for Blind Image Super-Resolution\](https://arxiv.org/abs/2203.04962)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/greatlog/UnpairedSR "GitHub - greatlog/UnpairedSR: This is an offical implementation of the CVPR2022's paper [Learning the Degradation Distribution for Blind Image Super-Resolution](https://arxiv.org/abs/2203.04962)")
*   Tags: Blind SR

**Deep Constrained Least Squares for Blind Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2202.07508](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2202.07508 "https://arxiv.org/abs/2202.07508")
*   Code:  [GitHub – Algolzw/DCLS: “Deep Constrained Least Squares for Blind Image Super-Resolution”, CVPR 2022.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Algolzw/DCLS "GitHub - Algolzw/DCLS:")
*   Tags: Blind SR

**Blind Image Super-resolution with Elaborate Degradation Modeling on Noise and Kernel**

*   Paper:  [https://arxiv.org/abs/2107.00986](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2107.00986 "https://arxiv.org/abs/2107.00986")
*   Code:  [https://github.com/zsyOAOA/BSRDM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zsyOAOA/BSRDM "https://github.com/zsyOAOA/BSRDM")
*   Tags: Blind SR

**Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2203.09195](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.09195 "https://arxiv.org/abs/2203.09195")
*   Code:  [https://github.com/csjliang/LDL](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/csjliang/LDL "https://github.com/csjliang/LDL")
*   Tags: Real SR

**Dual Adversarial Adaptation for Cross-Device Real-World Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2205.03524](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.03524 "https://arxiv.org/abs/2205.03524")
*   Code:  [GitHub - lonelyhope/DADA](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lonelyhope/DADA "GitHub - lonelyhope/DADA")
*   Tags: Real SR

**LAR-SR: A Local Autoregressive Model for Image Super-Resolution**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Guo_LAR-SR_A_Local_Autoregressive_Model_for_Image_Super-Resolution_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Texture-Based Error Analysis for Image Super-Resolution**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Magid_Texture-Based_Error_Analysis_for_Image_Super-Resolution_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Learning to Zoom Inside Camera Imaging Pipeline**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Learning_To_Zoom_Inside_Camera_Imaging_Pipeline_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Raw-to-Raw domain

**Task Decoupled Framework for Reference-Based Super-Resolution**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Reference-Based

**GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors**

*   Paper:  [https://arxiv.org/abs/2203.07319](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.07319 "https://arxiv.org/abs/2203.07319")
*   Code:  [GitHub – hejingwenhejingwen/GCFSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/hejingwenhejingwen/GCFSR "GitHub - hejingwenhejingwen/GCFSR")
*   Tags: Face SR

**A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution**

*   Paper:  [https://arxiv.org/abs/2203.09388](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.09388 "https://arxiv.org/abs/2203.09388")
*   Code:  [https://github.com/mjq11302010044/TATT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/mjq11302010044/TATT "https://github.com/mjq11302010044/TATT")
*   Tags: Text SR

**Learning Graph Regularization for Guided Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2203.14297](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.14297 "https://arxiv.org/abs/2203.14297")
*   Tags: Guided SR

**Transformer-empowered Multi-scale Contextual Matching and Aggregation for Multi-contrast MRI Super-resolution**

*   Paper:  [https://arxiv.org/abs/2203.13963](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.13963 "https://arxiv.org/abs/2203.13963")
*   Code:  [https://github.com/XAIMI-Lab/McMRSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/XAIMI-Lab/McMRSR "https://github.com/XAIMI-Lab/McMRSR")
*   Tags: MRI SR

**Discrete Cosine Transform Network for Guided Depth Map Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2104.06977](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2104.06977 "https://arxiv.org/abs/2104.06977")
*   Code:  [https://github.com/Zhaozixiang1228/GDSR-DCTNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Zhaozixiang1228/GDSR-DCTNet "https://github.com/Zhaozixiang1228/GDSR-DCTNet")
*   Tags: Guided Depth Map SR

**SphereSR: 360deg Image Super-Resolution With Arbitrary Projection via Continuous Spherical Image Representation**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Yoon_SphereSR_360deg_Image_Super-Resolution_With_Arbitrary_Projection_via_Continuous_Spherical_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**IM Deception: Grouped Information Distilling Super-Resolution Network**

*   Paper:  [https://arxiv.org/abs/2204.11463](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.11463 "https://arxiv.org/abs/2204.11463")
*   Tags: \[Workshop\], lightweight

**A Closer Look at Blind Super-Resolution: Degradation Models, Baselines, and Performance Upper Bounds**

*   Paper:  [https://arxiv.org/abs/2205.04910](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.04910 "https://arxiv.org/abs/2205.04910")
*   Code:  [https://github.com/WenlongZhang0517/CloserLookBlindSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/WenlongZhang0517/CloserLookBlindSR "https://github.com/WenlongZhang0517/CloserLookBlindSR")
*   Tags: \[Workshop\], Blind SR

### Burst/Multi-frame Super Resolution

**Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites**

*   Paper:  [https://arxiv.org/abs/2205.02031](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.02031 "https://arxiv.org/abs/2205.02031")
*   Code:  [GitHub – centreborelli/HDR-DSP-SR: Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/centreborelli/HDR-DSP-SR/ "GitHub - centreborelli/HDR-DSP-SR: Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites")
*   Tags: Self-Supervised, multi-exposure

### Video Super Resolution

**BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment**

*   Paper:  [https://arxiv.org/abs/2104.13371](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2104.13371 "https://arxiv.org/abs/2104.13371")
*   Code:  [GitHub – ckkelvinchan/BasicVSR\_PlusPlus: Official repository of “BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment”](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ckkelvinchan/BasicVSR_PlusPlus "GitHub - ckkelvinchan/BasicVSR_PlusPlus: Official repository of")

**Learning Trajectory-Aware Transformer for Video Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2204.04216](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.04216 "https://arxiv.org/abs/2204.04216")
*   Code:  [GitHub – researchmm/TTVSR: \[CVPR'22 Oral\] TTVSR: Learning Trajectory-Aware Transformer for Video Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/researchmm/TTVSR "GitHub - researchmm/TTVSR: [CVPR'22 Oral] TTVSR: Learning Trajectory-Aware Transformer for Video Super-Resolution")
*   Tags: Transformer

**Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling**

*   Paper:  [https://arxiv.org/abs/2204.07114](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.07114 "https://arxiv.org/abs/2204.07114")

**Investigating Tradeoffs in Real-World Video Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2111.12704](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.12704 "https://arxiv.org/abs/2111.12704")
*   Code:  [https://github.com/ckkelvinchan/RealBasicVSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ckkelvinchan/RealBasicVSR "https://github.com/ckkelvinchan/RealBasicVSR")
*   Tags: Real-world, RealBaiscVSR

**Memory-Augmented Non-Local Attention for Video Super-Resolution**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Memory-Augmented_Non-Local_Attention_for_Video_Super-Resolution_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Stable Long-Term Recurrent Video Super-Resolution**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Chiche_Stable_Long-Term_Recurrent_Video_Super-Resolution_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Reference-based Video Super-Resolution Using Multi-Camera Video Triplets**

*   Paper:  [https://arxiv.org/abs/2203.14537](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.14537 "https://arxiv.org/abs/2203.14537")
*   Code:  [https://github.com/codeslake/RefVSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/codeslake/RefVSR "https://github.com/codeslake/RefVSR")
*   Tags: Reference-based VSR

**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2204.10039](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.10039 "https://arxiv.org/abs/2204.10039")
*   Code:  [https://github.com/H-deep/Trans-SVSR/](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/H-deep/Trans-SVSR/ "https://github.com/H-deep/Trans-SVSR/")
*   Tags: Stereoscopic Video Super-Resolution

Image Rescaling – image scaling
-------------------------------

**Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence**

*   Paper:  [https://arxiv.org/abs/2203.00911](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.00911 "https://arxiv.org/abs/2203.00911")

**Faithful Extreme Rescaling via Generative Prior Reciprocated Invertible Representations**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_Faithful_Extreme_Rescaling_via_Generative_Prior_Reciprocated_Invertible_Representations_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Code:  [https://github.com/cszzx/GRAIN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/cszzx/GRAIN "https://github.com/cszzx/GRAIN")

Denoising – denoising
---------------------

### Image Denoising

**Self-Supervised Image Denoising via Iterative Data Refinement**

*   Paper:  [https://arxiv.org/abs/2111.14358](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.14358 "https://arxiv.org/abs/2111.14358")
*   Code:  [https://github.com/zhangyi-3/IDR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zhangyi-3/IDR "https://github.com/zhangyi-3/IDR")
*   Tags: Self-Supervised

**Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots**

*   Paper:  [https://arxiv.org/abs/2203.06967](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.06967 "https://arxiv.org/abs/2203.06967")
*   Code:  [https://github.com/demonsjin/Blind2Unblind](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/demonsjin/Blind2Unblind "https://github.com/demonsjin/Blind2Unblind")
*   Tags: Self-Supervised

**AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network**

*   Paper:  [https://arxiv.org/abs/2203.11799](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.11799 "https://arxiv.org/abs/2203.11799")
*   Code:  [https://github.com/wooseoklee4/AP-BSN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wooseoklee4/AP-BSN "https://github.com/wooseoklee4/AP-BSN")
*   Tags: Self-Supervised

**CVF-SID: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image**

*   Paper:  [https://arxiv.org/abs/2203.13009](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.13009 "https://arxiv.org/abs/2203.13009")
*   Code:  [GitHub – Reyhanehne/CVF-SID\_PyTorch](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Reyhanehne/CVF-SID_PyTorch "GitHub - Reyhanehne/CVF-SID_PyTorch")
*   Tags: Self-Supervised

**Noise Distribution Adaptive Self-Supervised Image Denoising Using Tweedie Distribution and Score Matching**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Noise_Distribution_Adaptive_Self-Supervised_Image_Denoising_Using_Tweedie_Distribution_and_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Self-Supervised

**Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images**

*   Paper:  [https://arxiv.org/abs/2206.01103](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.01103 "https://arxiv.org/abs/2206.01103")
*   Tags: Noise Modeling, Normalizing Flow

**Modeling sRGB Camera Noise with Normalizing Flows**

*   Paper:  [https://arxiv.org/abs/2206.00812](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.00812 "https://arxiv.org/abs/2206.00812")
*   Tags: Noise Modeling, Normalizing Flow

**Estimating Fine-Grained Noise Model via Contrastive Learning**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zou_Estimating_Fine-Grained_Noise_Model_via_Contrastive_Learning_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Noise Modeling, Constrastive Learning

**Multiple Degradation and Reconstruction Network for Single Image Denoising via Knowledge Distillation**

*   Paper:  [https://arxiv.org/abs/2204.13873](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.13873 "https://arxiv.org/abs/2204.13873")
*   Tags: \[Workshop\]

### Burst Denoising

**NAN: Noise-Aware NeRFs for Burst-Denoising**

*   Paper:  [https://arxiv.org/abs/2204.04668](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.04668 "https://arxiv.org/abs/2204.04668")
*   Tags: NeRFs

### Video Denoising

**Dancing under the stars: video denoising in starlight**

*   Paper:  [https://arxiv.org/abs/2204.04210](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.04210 "https://arxiv.org/abs/2204.04210")
*   Code:  [https://github.com/monakhova/starlight\_denoising/](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/monakhova/starlight_denoising/ "https://github.com/monakhova/starlight_denoising/")
*   Tags: video denoising in starlight

Deblurring – Deblurring
-----------------------

### Image Deblurring

**Learning to Deblur using Light Field Generated and Real Defocus Images**

*   Paper:  [https://arxiv.org/abs/2204.00367](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.00367 "https://arxiv.org/abs/2204.00367")
*   Code:  [https://github.com/lingyanruan/DRBNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lingyanruan/DRBNet "https://github.com/lingyanruan/DRBNet")
*   Tags: Defocus deblurring

**Pixel Screening Based Intermediate Correction for Blind Deblurring**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Pixel_Screening_Based_Intermediate_Correction_for_Blind_Deblurring_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Blind

**Deblurring via Stochastic Refinement**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Whang_Deblurring_via_Stochastic_Refinement_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**XYDeblur: Divide and Conquer for Single Image Deblurring**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Ji_XYDeblur_Divide_and_Conquer_for_Single_Image_Deblurring_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Unifying Motion Deblurring and Frame Interpolation with Events**

*   Paper:  [https://arxiv.org/abs/2203.12178](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.12178 "https://arxiv.org/abs/2203.12178")
*   Tags: event-based

**E-CIR: Event-Enhanced Continuous Intensity Recovery**

*   Paper:  [https://arxiv.org/abs/2203.01935](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.01935 "https://arxiv.org/abs/2203.01935")
*   Code:  [https://github.com/chensong1995/E-CIR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/chensong1995/E-CIR "https://github.com/chensong1995/E-CIR")
*   Tags: event-based

### Video Deblurring

**Multi-Scale Memory-Based Video Deblurring**

*   Paper:  [https://arxiv.org/abs/2203.01935](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.01935 "https://arxiv.org/abs/2203.01935")
*   Code:  [https://github.com/jibo27/MemDeblur](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jibo27/MemDeblur "https://github.com/jibo27/MemDeblur")

Deraining – deraining
---------------------

**Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**

*   Paper:  [https://arxiv.org/abs/2203.16931](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.16931 "https://arxiv.org/abs/2203.16931")
*   Code:  [https://github.com/yuyi-sd/Robust\_Rain\_Removal](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yuyi-sd/Robust_Rain_Removal "https://github.com/yuyi-sd/Robust_Rain_Removal")

**Unpaired Deep Image Deraining Using Dual Contrastive Learning**

*   Paper:  [https://arxiv.org/abs/2109.02973](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2109.02973 "https://arxiv.org/abs/2109.02973")
*   Tags: Contrastive Learning, Unpaired

**Unsupervised Deraining: Where Contrastive Learning Meets Self-similarity**

*   Paper:  [https://arxiv.org/abs/2203.11509](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.11509 "https://arxiv.org/abs/2203.11509")
*   Tags: Contrastive Learning, Unsupervised

**Dreaming To Prune Image Deraining Networks**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

Dehazing – to fog
-----------------

**Self-augmented Unpaired Image Dehazing via Density and Depth Decomposition**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Self-Augmented_Unpaired_Image_Dehazing_via_Density_and_Depth_Decomposition_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Code:  [https://github.com/YaN9-Y/D4](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/YaN9-Y/D4 "https://github.com/YaN9-Y/D4")
*   Tags: Unpaired

**Towards Multi-Domain Single Image Dehazing via Test-Time Training**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Multi-Domain_Single_Image_Dehazing_via_Test-Time_Training_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Image Dehazing Transformer With Transmission-Aware 3D Position Embedding**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Image_Dehazing_Transformer_With_Transmission-Aware_3D_Position_Embedding_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Physically Disentangled Intra- and Inter-Domain Adaptation for Varicolored Haze Removal**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Li_Physically_Disentangled_Intra-_and_Inter-Domain_Adaptation_for_Varicolored_Haze_Removal_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

Demoireing – Go moiré
---------------------

**Video Demoireing with Relation-Based Temporal Consistency**

*   Paper:  [https://arxiv.org/abs/2204.02957](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.02957 "https://arxiv.org/abs/2204.02957")
*   Code:  [https://github.com/CVMI-Lab/VideoDemoireing](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/CVMI-Lab/VideoDemoireing "https://github.com/CVMI-Lab/VideoDemoireing")

Frame Interpolation – frame insertion
-------------------------------------

**ST-MFNet: A Spatio-Temporal Multi-Flow Network for Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2111.15483](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.15483 "https://arxiv.org/abs/2111.15483")
*   Code:  [https://github.com/danielism97/ST-MFNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/danielism97/ST-MFNet "https://github.com/danielism97/ST-MFNet")

**Long-term Video Frame Interpolation via Feature Propagation**

*   Paper:  [https://arxiv.org/abs/2203.15427](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.15427 "https://arxiv.org/abs/2203.15427")

**Many-to-many Splatting for Efficient Video Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2204.03513](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.03513 "https://arxiv.org/abs/2204.03513")
*   Code:  [https://github.com/feinanshan/M2M\_VFI](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/feinanshan/M2M_VFI "https://github.com/feinanshan/M2M_VFI")

**Video Frame Interpolation with Transformer**

*   Paper:  [https://arxiv.org/abs/2205.07230](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.07230 "https://arxiv.org/abs/2205.07230")
*   Code:  [https://github.com/dvlab-research/VFIformer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/dvlab-research/VFIformer "https://github.com/dvlab-research/VFIformer")
*   Tags: Transformer

**Video Frame Interpolation Transformer**

*   Paper:  [https://arxiv.org/abs/2111.13817](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.13817 "https://arxiv.org/abs/2111.13817")
*   Code:  [https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer "https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer")
*   Tags: Transformer

**IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2205.14620](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.14620 "https://arxiv.org/abs/2205.14620")
*   Code:  [GitHub – ltkong218/IFRNet: IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation (CVPR 2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ltkong218/IFRNet "GitHub - ltkong218/IFRNet: IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation (CVPR 2022)")

**TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation**

*   Paper:  [https://arxiv.org/abs/2203.13859](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.13859 "https://arxiv.org/abs/2203.13859")
*   Tags: Event Camera

**Time Lens++: Event-based Frame Interpolation with Parametric Non-linear Flow and Multi-scale Fusion**

*   Paper:  [https://arxiv.org/abs/2203.17191](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.17191 "https://arxiv.org/abs/2203.17191")
*   Tags: Event-based

**Unifying Motion Deblurring and Frame Interpolation with Events**

*   Paper:  [https://arxiv.org/abs/2203.12178](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.12178 "https://arxiv.org/abs/2203.12178")
*   Tags: event-based

**Multi-encoder Network for Parameter Reduction of a Kernel-based Interpolation Architecture**

*   Paper:  [https://arxiv.org/abs/2205.06723](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.06723 "https://arxiv.org/abs/2205.06723")
*   Tags: \[Workshop\]

### Spatial-Temporal Video Super-Resolution

**RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2203.14186](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.14186 "https://arxiv.org/abs/2203.14186")
*   Code:  [https://github.com/llmpass/RSTT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/llmpass/RSTT "https://github.com/llmpass/RSTT")

**Spatial-Temporal Space Hand-in-Hand: Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning**

*   Paper:  [https://arxiv.org/abs/2205.05264](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.05264 "https://arxiv.org/abs/2205.05264")

**VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2206.04647](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.04647 "https://arxiv.org/abs/2206.04647")
*   Code:  [https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution "https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution")

Image Enhancement – ​​image enhancement
---------------------------------------

**AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-time Image Enhancement**

*   Paper:  [https://arxiv.org/abs/2204.13983](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.13983 "https://arxiv.org/abs/2204.13983")
*   Code:  [https://github.com/ImCharlesY/AdaInt](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ImCharlesY/AdaInt "https://github.com/ImCharlesY/AdaInt")

**Exposure Correction Model to Enhance Image Quality**

*   Paper:  [https://arxiv.org/abs/2204.10648](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.10648 "https://arxiv.org/abs/2204.10648")
*   Code:  [GitHub – yamand16/ExposureCorrection](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yamand16/ExposureCorrection "GitHub - yamad16/ExposureCorrection")
*   Tags: \[Workshop\]

### Low-Light Image Enhancement

**Abandoning the Bayer-Filter to See in the Dark**

*   Paper:  [https://arxiv.org/abs/2203.04042](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.04042 "https://arxiv.org/abs/2203.04042")
*   Code:  [https://github.com/TCL-AILab/Abandon\_Bayer-Filter\_See\_in\_the\_Dark](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark "https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark")

**Toward Fast, Flexible, and Robust Low-Light Image Enhancement**

*   Paper:  [https://arxiv.org/abs/2204.10137](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.10137 "https://arxiv.org/abs/2204.10137")
*   Code:  [GitHub – vis-opt-group/SCI: \[CVPR 2022\] This is the official code for the paper “Toward Fast, Flexible, and Robust Low-Light Image Enhancement”.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/vis-opt-group/SCI "GitHub - vis-opt-group/SCI: [CVPR 2022] This is the official code for the paper")

**Deep Color Consistent Network for Low-Light Image Enhancement**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Deep_Color_Consistent_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**SNR-Aware Low-Light Image Enhancement**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Code:  [https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance "https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance")

**URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

Image Harmonization – Image Harmonization
-----------------------------------------

**High-Resolution Image Harmonization via Collaborative Dual Transformationsg**

*   Paper:  [https://arxiv.org/abs/2109.06671](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2109.06671 "https://arxiv.org/abs/2109.06671")
*   Code:  [GitHub – bcmi/CDTNet-High-Resolution-Image-Harmonization: \[CVPR 2022\] We unify pixel-to-pixel transformation and color-to-color transformation in a coherent framework for high-resolution image harmonization. We also release 100 high-resolution real composite images for evaluation.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization "GitHub - bcmi/CDTNet-High-Resolution-Image-Harmonization: [CVPR 2022] We unify pixel-to-pixel transformation and color-to-color transformation in a coherent framework for high-resolution image harmonization. We also release 100 high- resolution real composite images for evaluation.")

**SCS-Co: Self-Consistent Style Contrastive Learning for Image Harmonization**

*   Paper:  [https://arxiv.org/abs/2204.13962](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.13962 "https://arxiv.org/abs/2204.13962")
*   Code:  [GitHub – YCHang686/SCS-Co-CVPR2022: SCS-Co: Self-Consistent Style Contrastive Learning for Image Harmonization (CVPR 2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/YCHang686/SCS-Co-CVPR2022 "GitHub - YCHang686/SCS-Co-CVPR2022: SCS-Co: Self-Consistent Style Contrastive Learning for Image Harmonization (CVPR 2022)")

**Deep Image-based Illumination Harmonization**

*   Paper:  [https://arxiv.org/abs/2108.00150](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2108.00150 "https://arxiv.org/abs/2108.00150")
*   Dataset:  [https://github.com/zhongyunbao/Dataset](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zhongyunbao/Dataset "https://github.com/zhongyunbao/Dataset")

Image Completion/Inpainting – image restoration
-----------------------------------------------

**Bridging Global Context Interactions for High-Fidelity Image Completion**

*   Paper:  [https://arxiv.org/abs/2104.00845](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2104.00845 "https://arxiv.org/abs/2104.00845")
*   Code:  [https://github.com/lyndonzheng/TFill](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lyndonzheng/TFill "https://github.com/lyndonzheng/TFill")

**Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding**

*   Paper:  [https://arxiv.org/abs/2203.00867](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.00867 "https://arxiv.org/abs/2203.00867")
*   Code:  [GitHub – DQiaole/ZITS\_inpainting: Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding (CVPR2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DQiaole/ZITS_inpainting "GitHub - DQiaole/ZITS_inpainting: Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding (CVPR2022)")

**MISF: Multi-level Interactive Siamese Filtering for High-Fidelity Image Inpainting**

*   Paper:  [https://arxiv.org/abs/2203.06304](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.06304 "https://arxiv.org/abs/2203.06304")
*   Code:  [GitHub – tsingqguo/misf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/tsingqguo/misf "GitHub - tsingqguo/misf")

**MAT: Mask-Aware Transformer for Large Hole Image Inpainting**

*   Paper:  [https://arxiv.org/abs/2203.15270](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.15270 "https://arxiv.org/abs/2203.15270")
*   Code:  [GitHub – fenglinglwb/MAT: MAT: Mask-Aware Transformer for Large Hole Image Inpainting](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fenglinglwb/MAT "GitHub - fenglinglwb/MAT: MAT: Mask-Aware Transformer for Large Hole Image Inpainting")

**Reduce Information Loss in Transformers for Pluralistic Image Inpainting**

*   Paper:  [https://arxiv.org/abs/2205.05076](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.05076 "https://arxiv.org/abs/2205.05076")
*   Code:  [GitHub – liuqk3/PUT: Paper 'Reduce Information Loss in Transformers for Pluralistic Image Inpainting' in CVPR2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/liuqk3/PUT "GitHub - liuqk3/PUT: Paper 'Reduce Information Loss in Transformers for Pluralistic Image Inpainting' in CVPR2022")

**RePaint: Inpainting using Denoising Diffusion Probabilistic Models**

*   Paper:  [https://arxiv.org/abs/2201.09865](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2201.09865 "https://arxiv.org/abs/2201.09865")
*   Code:  [GitHub – andreas128/RePaint: Official PyTorch Code and Models of “RePaint: Inpainting using Denoising Diffusion Probabilistic Models”, CVPR 2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/andreas128/RePaint "GitHub - andreas128/RePaint: Official PyTorch Code and Models of")
*   Tags: DDPM

**Dual-Path Image Inpainting With Auxiliary GAN Inversion**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Dual-Path_Image_Inpainting_With_Auxiliary_GAN_Inversion_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**SaiNet: Stereo aware inpainting behind objects with generative networks**

*   Paper:  [https://arxiv.org/abs/2205.07014](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.07014 "https://arxiv.org/abs/2205.07014")
*   Tags: \[Workshop\]

### Video Inpainting

**Towards An End-to-End Framework for Flow-Guided Video Inpainting**

*   Paper:  [https://arxiv.org/abs/2204.02663](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.02663 "https://arxiv.org/abs/2204.02663")
*   Code:  [https://github.com/MCG-NKU/E2FGVI](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/MCG-NKU/E2FGVI "https://github.com/MCG-NKU/E2FGVI")

**The DEVIL Is in the Details: A Diagnostic Evaluation Benchmark for Video Inpainting**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Szeto_The_DEVIL_Is_in_the_Details_A_Diagnostic_Evaluation_Benchmark_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Code:  [GitHub – MichiganCOG/devil](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/MichiganCOG/devil "GitHub - MichiganCOG/devil")

**DLFormer: Discrete Latent Transformer for Video Inpainting**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Ren_DLFormer_Discrete_Latent_Transformer_for_Video_Inpainting_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Inertia-Guided Flow Completion and Style Fusion for Video Inpainting**

*   Paper:  [https://openaccess.thecvf.com/content/CVPR2022/html/Zhang\_Inertia-Guided\_Flow\_Completion\_and\_Style\_Fusion\_for\_Video\_Inpainting\_CVPR\_2022\_paper.htmll](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Inertia-Guided_Flow_Completion_and_Style_Fusion_for_Video_Inpainting_CVPR_2022_paper.htmll "https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Inertia-Guided_Flow_Completion_and_Style_Fusion_for_Video_Inpainting_CVPR_2022_paper.htmll")

Image Matting – image matting
-----------------------------

**MatteFormer: Transformer-Based Image Matting via Prior-Tokens**

*   Paper:  [https://arxiv.org/abs/2203.15662](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.15662 "https://arxiv.org/abs/2203.15662")
*   Code:  [https://github.com/webtoon/matteformer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/webtoon/matteformer "https://github.com/webtoon/matteformer")

**Human Instance Matting via Mutual Guidance and Multi-Instance Refinement**

*   Paper:  [https://arxiv.org/abs/2205.10767](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.10767 "https://arxiv.org/abs/2205.10767")
*   Code:  [GitHub – nowsyn/InstMatt: Official repository for Instance Human Matting via Mutual Guidance and Multi-Instance Refinement](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/nowsyn/InstMatt "GitHub - nowsyn/InstMatt: Official repository for Instance Human Matting via Mutual Guidance and Multi-Instance Refinement")

**Boosting Robustness of Image Matting with Context Assembly and Strong Data Augmentation**

*   Paper:  [https://arxiv.org/abs/2201.06889](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2201.06889 "https://arxiv.org/abs/2201.06889")

Shadow Removal – shadow removal
-------------------------------

**Bijective Mapping Network for Shadow Removal**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Bijective_Mapping_Network_for_Shadow_Removal_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

Relighting
----------

**Face Relighting with Geometrically Consistent Shadows**

*   Paper:  [https://arxiv.org/abs/2203.16681](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.16681 "https://arxiv.org/abs/2203.16681")
*   Code:  [GitHub – andrewhou1/GeomConsistentFR: Official Code for Face Relighting with Geometrically Consistent Shadows (CVPR 2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/andrewhou1/GeomConsistentFR "GitHub - andrewhou1/GeomConsistentFR: Official Code for Face Relighting with Geometrically Consistent Shadows (CVPR 2022)")
*   Tags: Face Relighting

**SIMBAR: Single Image-Based Scene Relighting For Effective Data Augmentation For Automated Driving Vision Tasks**

*   Paper:  [https://arxiv.org/abs/2204.00644](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.00644 "https://arxiv.org/abs/2204.00644")

Image Stitching – image stitching
---------------------------------

**Deep Rectangling for Image Stitching: A Learning Baseline**

*   Paper:  [https://arxiv.org/abs/2203.03831](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.03831 "https://arxiv.org/abs/2203.03831")
*   Code:  [https://github.com/nie-lang/DeepRectangling](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/nie-lang/DeepRectangling "https://github.com/nie-lang/DeepRectangling")

**Automatic Color Image Stitching Using Quaternion Rank-1 Alignment**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Li_Automatic_Color_Image_Stitching_Using_Quaternion_Rank-1_Alignment_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Geometric Structure Preserving Warp for Natural Image Stitching**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Du_Geometric_Structure_Preserving_Warp_for_Natural_Image_Stitching_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

Image Compression – image compression
-------------------------------------

**Neural Data-Dependent Transform for Learned Image Compression**

*   Paper:  [https://arxiv.org/abs/2203.04963v1](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.04963v1 "https://arxiv.org/abs/2203.04963v1")

**The Devil Is in the Details: Window-based Attention for Image Compression**

*   Paper:  [https://arxiv.org/abs/2203.08450](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.08450 "https://arxiv.org/abs/2203.08450")
*   Code:  [https://github.com/Googolxx/STF](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Googolxx/STF "https://github.com/Googolxx/STF")

**ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding**

*   Paper:  [https://arxiv.org/abs/2203.10886](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.10886 "https://arxiv.org/abs/2203.10886")

**Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression**

*   Paper:  [https://arxiv.org/abs/2203.10897](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.10897 "https://arxiv.org/abs/2203.10897")
*   Code:  [GitHub – xiaosu-zhu/McQuic: Repository of CVPR'22 paper “Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression”](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/xiaosu-zhu/McQuic "GitHub - xiaosu-zhu/McQuic: Repository of CVPR'22 paper")

**DPICT: Deep Progressive Image Compression Using Trit-Planes**

*   Paper:  [https://arxiv.org/abs/2112.06334](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.06334 "https://arxiv.org/abs/2112.06334")
*   Code:  [https://github.com/jaehanlee-mcl/DPICT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jaehanlee-mcl/DPICT "https://github.com/jaehanlee-mcl/DPICT")

**Joint Global and Local Hierarchical Priors for Learned Image Compression**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Joint_Global_and_Local_Hierarchical_Priors_for_Learned_Image_Compression_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**LC-FDNet: Learned Lossless Image Compression With Frequency Decomposition Network**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Rhee_LC-FDNet_Learned_Lossless_Image_Compression_With_Frequency_Decomposition_Network_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Practical Learned Lossless JPEG Recompression with Multi-Level Cross-Channel Entropy Model in the DCT Domain**

*   Paper:  [https://arxiv.org/abs/2203.16357](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.16357 "https://arxiv.org/abs/2203.16357")
*   Tags: Compress JPEG

**SASIC: Stereo Image Compression With Latent Shifts and Stereo Attention**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Wodlinger_SASIC_Stereo_Image_Compression_With_Latent_Shifts_and_Stereo_Attention_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Stereo Image Compression

**Deep Stereo Image Compression via Bi-Directional Coding**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Lei_Deep_Stereo_Image_Compression_via_Bi-Directional_Coding_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Stereo Image Compression

**Learning Based Multi-Modality Image and Video Compression**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Lu_Learning_Based_Multi-Modality_Image_and_Video_Compression_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**PO-ELIC: Perception-Oriented Efficient Learned Image Coding**

*   Paper:  [https://arxiv.org/abs/2205.14501](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.14501 "https://arxiv.org/abs/2205.14501")
*   Tags: \[Workshop\]

### Video Compression

**Coarse-to-fine Deep Video Coding with Hyperprior-guided Mode Prediction**

*   Paper:  [https://arxiv.org/abs/2206.07460](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.07460 "https://arxiv.org/abs/2206.07460")

**LSVC: A Learning-Based Stereo Video Compression Framework**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Chen_LSVC_A_Learning-Based_Stereo_Video_Compression_Framework_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Tags: Stereo Video Compression

**Enhancing VVC with Deep Learning based Multi-Frame Post-Processing**

*   Paper:  [https://arxiv.org/abs/2205.09458](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.09458 "https://arxiv.org/abs/2205.09458")
*   Tags: \[Workshop\]

Image Quality Assessment – ​​image quality assessment
-----------------------------------------------------

**Personalized Image Aesthetics Assessment with Rich Attributes**

*   Paper:  [https://arxiv.org/abs/2203.16754](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.16754 "https://arxiv.org/abs/2203.16754")
*   Tags: Aesthetics Assessment

**Incorporating Semi-Supervised and Positive-Unlabeled Learning for Boosting Full Reference Image Quality Assessment**

*   Paper:  [https://arxiv.org/abs/2204.08763](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.08763 "https://arxiv.org/abs/2204.08763")
*   Code:  [GitHub – happycaoyue/JSPL](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/happycaoyue/JSPL "GitHub - happycaoyue/JSPL")
*   Tags: FR-IQA

**SwinIQA: Learned Swin Distance for Compressed Image Quality Assessment**

*   Paper:  [https://arxiv.org/abs/2205.04264](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.04264 "https://arxiv.org/abs/2205.04264")
*   Tags: \[Workshop\], compressed IQA

Image Decomposition
-------------------

**PIE-Net: Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Das_PIE-Net_Photometric_Invariant_Edge_Guided_Network_for_Intrinsic_Image_Decomposition_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Code:  [GitHub – Morpheus3000/PIE-Net: Official model and network release for my CVPR2022 paper.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Morpheus3000/PIE-Net "GitHub - Morpheus3000/PIE-Net: Official model and network release for my CVPR2022 paper.")

**Deformable Sprites for Unsupervised Video Decomposition**

*   Paper:  [https://arxiv.org/abs/2204.07151](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.07151 "https://arxiv.org/abs/2204.07151")
*   Code:  [https://github.com/vye16/deformable-sprites](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/vye16/deformable-sprites "https://github.com/vye16/deformable-sprites")

Style Transfer – style transfer
-------------------------------

**CLIPstyler: Image Style Transfer with a Single Text Condition**

*   Paper:  [https://arxiv.org/abs/2112.00374](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.00374 "https://arxiv.org/abs/2112.00374")
*   Code:  [GitHub – cyclomon/CLIPstyler: Official Pytorch implementation of “CLIPstyler: Image Style Transfer with a Single Text Condition” (CVPR 2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/cyclomon/CLIPstyler "GitHub - cyclomon/CLIPstyler: Official Pytorch implementation of")
*   Tags: CLIP

**Style-ERD: Responsive and Coherent Online Motion Style Transfer**

*   Paper:  [https://arxiv.org/abs/2203.02574](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.02574 "https://arxiv.org/abs/2203.02574")
*   Code:  [GitHub – tianxintao/Online-Motion-Style-Transfer: Code for the CVPR 2022 Paper – Style-ERD: Responsive and Coherent Online Motion Style Transfer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/tianxintao/Online-Motion-Style-Transfer "GitHub - tianxintao/Online-Motion-Style-Transfer: Code for the CVPR 2022 Paper - Style-ERD: Responsive and Coherent Online Motion Style Transfer")

**Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization**

*   Paper:  [https://arxiv.org/abs/2203.07740](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.07740 "https://arxiv.org/abs/2203.07740")
*   Code:  [GitHub – YBZh/EFDM: Official PyTorch codes of CVPR2022 Oral: Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/YBZh/EFDM "GitHub - YBZh/EFDM: Official PyTorch codes of CVPR2022 Oral: Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization")

**Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer**

*   Paper:  [https://arxiv.org/abs/2203.13248](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.13248 "https://arxiv.org/abs/2203.13248")
*   Code:  [https://github.com/williamyang1991/DualStyleGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/williamyang1991/DualStyleGAN "https://github.com/williamyang1991/DualStyleGAN")

**Industrial Style Transfer with Large-scale Geometric Warping and Content Preservation**

*   Paper:  [https://arxiv.org/abs/2203.12835](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.12835 "https://arxiv.org/abs/2203.12835")
*   Code:  [https://github.com/jcyang98/InST](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jcyang98/InST "https://github.com/jcyang98/InST")

**StyTr2: Image Style Transfer With Transformers**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Deng_StyTr2_Image_Style_Transfer_With_Transformers_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**PCA-Based Knowledge Distillation Towards Lightweight and Content-Style Balanced Photorealistic Style Transfer Models**

*   Paper:  [https://arxiv.org/abs/2203.13452](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.13452 "https://arxiv.org/abs/2203.13452")
*   Code:  [GitHub – chiutaiyin/PCA-Knowledge-Distillation: PCA-based knowledge distillation towards lightweight and content-style balanced photorealistic style transfer models](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/chiutaiyin/PCA-Knowledge-Distillation "GitHub - chiutaiyin/PCA-Knowledge-Distillation: PCA-based knowledge distillation towards lightweight and content-style balanced photorealistic style transfer models")

Image Editing – image editing
-----------------------------

**High-Fidelity GAN Inversion for Image Attribute Editing**

*   Paper:  [https://arxiv.org/abs/2109.06590](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2109.06590 "https://arxiv.org/abs/2109.06590")
*   Code:  [https://github.com/Tengfei-Wang/HFGI](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Tengfei-Wang/HFGI "https://github.com/Tengfei-Wang/HFGI")

**Style Transformer for Image Inversion and Editing**

*   Paper:  [https://arxiv.org/abs/2203.07932](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.07932 "https://arxiv.org/abs/2203.07932")
*   Code:  [https://github.com/sapphire497/style-transformer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sapphire497/style-transformer "https://github.com/sapphire497/style-transformer")

**HairCLIP: Design Your Hair by Text and Reference Image**

*   Paper:  [https://arxiv.org/abs/2112.05142](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.05142 "https://arxiv.org/abs/2112.05142")
*   Code:  [GitHub – wty-ustc/HairCLIP: \[CVPR 2022\] HairCLIP: Design Your Hair by Text and Reference Image](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wty-ustc/HairCLIP "GitHub - wty-ustc/HairCLIP: [CVPR 2022] HairCLIP: Design Your Hair by Text and Reference Image")
*   Tags: CLIP

**HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing**

*   Paper:  [https://arxiv.org/abs/2111.15666](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.15666 "https://arxiv.org/abs/2111.15666")
*   Code:  [GitHub – yuval-alaluf/hyperstyle: Official Implementation for “HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing” (CVPR 2022) https://arxiv.org/abs/2111.15666](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yuval-alaluf/hyperstyle "GitHub - yuval-alaluf/hyperstyle: Official Implementation for")

**Blended Diffusion for Text-driven Editing of Natural Images**

*   Paper:  [https://arxiv.org/abs/2111.14818](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.14818 "https://arxiv.org/abs/2111.14818")
*   Code:  [GitHub – omriav/blended-diffusion: Official implementation for “Blended Diffusion for Text-driven Editing of Natural Images” \[CVPR 2022\]](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/omriav/blended-diffusion "GitHub - omriav/blended-diffusion: Official implementation for")
*   Tags: CLIP, Diffusion Model

**FlexIT: Towards Flexible Semantic Image Translation**

*   Paper:  [https://arxiv.org/abs/2203.04705](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.04705 "https://arxiv.org/abs/2203.04705")

**SemanticStyleGAN: Learning Compositonal Generative Priors for Controllable Image Synthesis and Editing**

*   Paper:  [https://arxiv.org/abs/2112.02236](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.02236 "https://arxiv.org/abs/2112.02236")

**SketchEdit: Mask-Free Local Image Manipulation with Partial Sketches**

*   Paper:  [https://arxiv.org/abs/2111.15078](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.15078 "https://arxiv.org/abs/2111.15078")
*   Code:  [https://github.com/zengxianyu/sketchedit](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zengxianyu/sketchedit "https://github.com/zengxianyu/sketchedit")

**TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing**

*   Paper:  [https://arxiv.org/abs/2203.17266](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.17266 "https://arxiv.org/abs/2203.17266")
*   Code:  [GitHub – BillyXYB/TransEditor: \[CVPR 2022\] TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/BillyXYB/TransEditor "GitHub - BillyXYB/TransEditor: [CVPR 2022] TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing")

**HyperInverter: Improving StyleGAN Inversion via Hypernetwork**

*   Paper:  [https://arxiv.org/abs/2112.00719](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.00719 "https://arxiv.org/abs/2112.00719")
*   Code:  [GitHub – VinAIResearch/HyperInverter: HyperInverter: Improving StyleGAN Inversion via Hypernetwork (CVPR 2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/VinAIResearch/HyperInverter "GitHub - VinAIResearch/HyperInverter: HyperInverter: Improving StyleGAN Inversion via Hypernetwork (CVPR 2022)")

**Spatially-Adaptive Multilayer Selection for GAN Inversion and Editing**

*   Paper:  [https://arxiv.org/abs/2206.08357](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.08357 "https://arxiv.org/abs/2206.08357")
*   Code:  [GitHub – adobe-research/sam\_inversion: \[CVPR 2022\] GAN inversion and editing with spatially-adaptive multiple latent layers](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/adobe-research/sam_inversion "GitHub - adobe-research/sam_inversion: [CVPR 2022] GAN inversion and editing with spatially-adaptive multiple latent layers")

**Brain-Supervised Image Editing**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Davis_Brain-Supervised_Image_Editing_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**SpaceEdit: Learning a Unified Editing Space for Open-Domain Image Color Editing**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Shi_SpaceEdit_Learning_a_Unified_Editing_Space_for_Open-Domain_Image_Color_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**M3L: Language-based Video Editing via Multi-Modal Multi-Level Transformers**

*   Paper:  [https://arxiv.org/abs/2104.01122](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2104.01122 "https://arxiv.org/abs/2104.01122")

Image Generation/Synthesis / Image-to-Image Translation – Image Generation/Synthesis/Translation
------------------------------------------------------------------------------------------------

### Text-to-Image / Text Guided / Multi-Modal

**Text to Image Generation with Semantic-Spatial Aware GAN**

*   Paper:  [https://arxiv.org/abs/2104.00567](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2104.00567 "https://arxiv.org/abs/2104.00567")
*   Code:  [GitHub – wtliao/text2image: Text to Image Generation with Semantic-Spatial Aware GAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wtliao/text2image "GitHub - wtliao/text2image: Text to Image Generation with Semantic-Spatial Aware GAN")

**LAFITE: Towards Language-Free Training for Text-to-Image Generation**

*   Paper:  [https://arxiv.org/abs/2111.13792](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.13792 "https://arxiv.org/abs/2111.13792")
*   Code:  [https://github.com/drboog/Lafite](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/drboog/Lafite "https://github.com/drboog/Lafite")

**DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2008.05865](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2008.05865 "https://arxiv.org/abs/2008.05865")
*   Code:  [GitHub – tobran/DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis (CVPR2022 oral)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/tobran/DF-GAN "GitHub - tobran/DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis (CVPR2022 oral)")

**StyleT2I: Toward Compositional and High-Fidelity Text-to-Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2203.15799](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.15799 "https://arxiv.org/abs/2203.15799")
*   Code:  [https://github.com/zhihengli-UR/StyleT2I](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zhihengli-UR/StyleT2I "https://github.com/zhihengli-UR/StyleT2I")

**DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation**

*   Paper:  [https://arxiv.org/abs/2110.02711](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2110.02711 "https://arxiv.org/abs/2110.02711")
*   Code:  [GitHub – gwang-kim/DiffusionCLIP: \[CVPR 2022\] Official PyTorch Implementation for DiffusionCLIP: Text-guided Image Manipulation Using Diffusion Models](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/gwang-kim/DiffusionCLIP "GitHub - gwang-kim/DiffusionCLIP: [CVPR 2022] Official PyTorch Implementation for DiffusionCLIP: Text-guided Image Manipulation Using Diffusion Models")

**Predict, Prevent, and Evaluate: Disentangled Text-Driven Image Manipulation Empowered by Pre-Trained Vision-Language Model**

*   Paper:  [https://arxiv.org/abs/2111.13333](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.13333 "https://arxiv.org/abs/2111.13333")
*   Code:  [GitHub – zipengxuc/PPE-Pytorch: Pytorch Implementation for CVPR'2022 paper ✨ “Predict, Prevent, and Evaluate: Disentangled Text-Driven Image Manipulation Empowered by Pre-Trained Vision-Language Model”](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zipengxuc/PPE-Pytorch "GitHub - zipengxuc/PPE-Pytorch: Pytorch Implementation for CVPR'2022 paper ✨")

**Sound-Guided Semantic Image Manipulation**

*   Paper:  [https://arxiv.org/abs/2112.00007](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.00007 "https://arxiv.org/abs/2112.00007")
*   Code:  [https://github.com/kuai-lab/sound-guided-semantic-image-manipulation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/kuai-lab/sound-guided-semantic-image-manipulation "https://github.com/kuai-lab/sound-guided-semantic-image-manipulation")

**ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation**

*   Paper:  [https://arxiv.org/abs/2204.04428](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.04428 "https://arxiv.org/abs/2204.04428")

**Text-to-Image Synthesis Based on Object-Guided Joint-Decoding Transformer**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Text-to-Image_Synthesis_Based_on_Object-Guided_Joint-Decoding_Transformer_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Vector Quantized Diffusion Model for Text-to-Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2111.14822](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.14822 "https://arxiv.org/abs/2111.14822")

**AnyFace: Free-style Text-to-Face Synthesis and Manipulation**

*   Paper:  [https://arxiv.org/abs/2203.15334](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.15334 "https://arxiv.org/abs/2203.15334")

### Image-to-Image / Image Guided

**Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation**

*   Paper:  [https://arxiv.org/abs/2203.12707](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.12707 "https://arxiv.org/abs/2203.12707")
*   Code:  [https://github.com/batmanlab/MSPC](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/batmanlab/MSPC "https://github.com/batmanlab/MSPC")

**A Style-aware Discriminator for Controllable Image Translation**

*   Paper:  [https://arxiv.org/abs/2203.15375](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.15375 "https://arxiv.org/abs/2203.15375")
*   Code:  [https://github.com/kunheek/style-aware-discriminator](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/kunheek/style-aware-discriminator "https://github.com/kunheek/style-aware-discriminator")

**QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation**

*   Paper:  [https://arxiv.org/abs/2203.08483](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.08483 "https://arxiv.org/abs/2203.08483")
*   Code:  [GitHub – sapphire497/query-selected-attention: Official implementation for “QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation” (CVPR 2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sapphire497/query-selected-attention "GitHub - sapphire497/query-selected-attention: Official implementation for")

**InstaFormer: Instance-Aware Image-to-Image Translation with Transformer**

*   Paper:  [https://arxiv.org/abs/2203.16248](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.16248 "https://arxiv.org/abs/2203.16248")

**Marginal Contrastive Correspondence for Guided Image Generation**

*   Paper:  [https://arxiv.org/abs/2204.00442](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.00442 "https://arxiv.org/abs/2204.00442")
*   Code:  [GitHub – fnzhan/UNITE: Unbalanced Feature Transport for Exemplar-based Image Translation \[CVPR 2021\] and Marginal Contrastive Correspondence for Guided Image Generation \[CVPR 2022\]](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fnzhan/UNITE "GitHub - fnzhan/UNITE: Unbalanced Feature Transport for Exemplar-based Image Translation [CVPR 2021] and Marginal Contrastive Correspondence for Guided Image Generation [CVPR 2022]")

**Unsupervised Image-to-Image Translation with Generative Prior**

*   Paper:  [https://arxiv.org/abs/2204.03641](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.03641 "https://arxiv.org/abs/2204.03641")
*   Code:  [https://github.com/williamyang1991/GP-UNIT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/williamyang1991/GP-UNIT "https://github.com/williamyang1991/GP-UNIT")

**Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks**

*   Paper:  [https://arxiv.org/abs/2203.01532](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.01532 "https://arxiv.org/abs/2203.01532")
*   Code:  [GitHub – jcy132/Hneg\_SRC: Official Pytorch implementation of “Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks” (CVPR 2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jcy132/Hneg_SRC "GitHub - jcy132/Hneg_SRC: Official Pytorch implementation of")

**Neural Texture Extraction and Distribution for Controllable Person Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2204.06160](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.06160 "https://arxiv.org/abs/2204.06160")
*   Code:  [GitHub – RenYurui/Neural-Texture-Extraction-Distribution: The PyTorch implementation for paper “Neural Texture Extraction and Distribution for Controllable Person Image Synthesis” (CVPR2022 Oral)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/RenYurui/Neural-Texture-Extraction-Distribution "GitHub - RenYurui/Neural-Texture-Extraction-Distribution: The PyTorch implementation for paper")

**Unpaired Cartoon Image Synthesis via Gated Cycle Mapping**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Men_Unpaired_Cartoon_Image_Synthesis_via_Gated_Cycle_Mapping_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Day-to-Night Image Synthesis for Training Nighttime Neural ISPs**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Punnappurath_Day-to-Night_Image_Synthesis_for_Training_Nighttime_Neural_ISPs_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")
*   Code:  [GitHub – SamsungLabs/day-to-night](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/SamsungLabs/day-to-night "GitHub - SamsungLabs/day-to-night")

**Alleviating Semantics Distortion in Unsupervised Low-Level Image-to-Image Translation via Structure Consistency Constraint**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Alleviating_Semantics_Distortion_in_Unsupervised_Low-Level_Image-to-Image_Translation_via_Structure_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Wavelet Knowledge Distillation: Towards Efficient Image-to-Image Translation**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Wavelet_Knowledge_Distillation_Towards_Efficient_Image-to-Image_Translation_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Self-Supervised Dense Consistency Regularization for Image-to-Image Translation**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Ko_Self-Supervised_Dense_Consistency_Regularization_for_Image-to-Image_Translation_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Drop the GAN: In Defense of Patches Nearest Neighbors as Single Image Generative Model**

*   Paper:  [https://arxiv.org/abs/2103.15545](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2103.15545 "https://arxiv.org/abs/2103.15545")
*   Project Web:  [“Drop The GAN: In Defense of Patch Nearest Neighbors as as Single Image Generative Models](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.wisdom.weizmann.ac.il/~vision/gpnn/)
*   Tags: image manipulation

**HairMapper: Removing Hair From Portraits Using GANs**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Wu_HairMapper_Removing_Hair_From_Portraits_Using_GANs_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

### Others for image generation

**Attribute Group Editing for Reliable Few-shot Image Generation**

*   Paper:  [https://arxiv.org/abs/2203.08422](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.08422 "https://arxiv.org/abs/2203.08422")
*   Code:  [https://github.com/UniBester/AGE](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/UniBester/AGE "https://github.com/UniBester/AGE")

**Modulated Contrast for Versatile Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2203.09333](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.09333 "https://arxiv.org/abs/2203.09333")
*   Code:  [GitHub – fnzhan/MoNCE: Modulated Contrast for Versatile Image Synthesis \[CVPR 2022\]](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fnzhan/MoNCE "GitHub - fnzhan/MoNCE: Modulated Contrast for Versatile Image Synthesis [CVPR 2022]")

**Interactive Image Synthesis with Panoptic Layout Generation**

*   Paper:  [https://arxiv.org/abs/2203.02104](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.02104 "https://arxiv.org/abs/2203.02104")

**Autoregressive Image Generation using Residual Quantization**

*   Paper:  [https://arxiv.org/abs/2203.01941](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.01941 "https://arxiv.org/abs/2203.01941")
*   Code:  [GitHub – lucidrains/RQ-Transformer: Implementation of RQ Transformer, proposed in the paper “Autoregressive Image Generation using Residual Quantization”](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lucidrains/RQ-Transformer "GitHub - lucidrains/RQ-Transformer: Implementation of RQ Transformer, proposed in the paper")

**Dynamic Dual-Output Diffusion Models**

*   Paper:  [https://arxiv.org/abs/2203.04304](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.04304 "https://arxiv.org/abs/2203.04304")

**Exploring Dual-task Correlation for Pose Guided Person Image Generation**

*   Paper:  [https://arxiv.org/abs/2203.02910](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.02910 "https://arxiv.org/abs/2203.02910")
*   Code:  [GitHub – PangzeCheung/Dual-task-Pose-Transformer-Network: \[CVPR 2022\] Exploring Dual-task Correlation for Pose Guided Person Image Generation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network "GitHub - PangzeCheung/Dual-task-Pose-Transformer-Network: [CVPR 2022] Exploring Dual-task Correlation for Pose Guided Person Image Generation")

**StyleSwin: Transformer-based GAN for High-resolution Image Generation**

*   Paper:  [https://arxiv.org/abs/2112.10762](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.10762 "https://arxiv.org/abs/2112.10762")
*   Code:  [GitHub – microsoft/StyleSwin: \[CVPR 2022\] StyleSwin: Transformer-based GAN for High-resolution Image Generation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/microsoft/StyleSwin "GitHub - microsoft/StyleSwin: [CVPR 2022] StyleSwin: Transformer-based GAN for High-resolution Image Generation")

**Semantic-shape Adaptive Feature Modulation for Semantic Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2203.16898](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.16898 "https://arxiv.org/abs/2203.16898")
*   Code:  [GitHub – cszy98/SAFM: Semantic-shape Adaptive Feature Modulation for Semantic Image Synthesis (CVPR2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/cszy98/SAFM "GitHub - cszy98/SAFM: Semantic-shape Adaptive Feature Modulation for Semantic Image Synthesis (CVPR2022)")

**Arbitrary-Scale Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2204.02273](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.02273 "https://arxiv.org/abs/2204.02273")
*   Code:  [https://github.com/vglsd/ScaleParty](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/vglsd/ScaleParty "https://github.com/vglsd/ScaleParty")

**InsetGAN for Full-Body Image Generation**

*   Paper:  [https://arxiv.org/abs/2203.07293](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.07293 "https://arxiv.org/abs/2203.07293")

**HairMapper: Removing Hair from Portraits Using GANs**

*   Paper:  [http://www.cad.zju.edu.cn/home/jin/cvpr2022/HairMapper.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=http://www.cad.zju.edu.cn/home/jin/cvpr2022/HairMapper.pdf "http://www.cad.zju.edu.cn/home/jin/cvpr2022/HairMapper.pdf")
*   Code:  [https://github.com/oneThousand1000/non-hair-FFHQ](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/oneThousand1000/non-hair-FFHQ "https://github.com/oneThousand1000/non-hair-FFHQ")

**OSSGAN: Open-Set Semi-Supervised Image Generation**

*   Paper:  [https://arxiv.org/abs/2204.14249](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.14249 "https://arxiv.org/abs/2204.14249")
*   Code:  [https://github.com/raven38/OSSGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/raven38/OSSGAN "https://github.com/raven38/OSSGAN")

**Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2204.02854](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.02854 "https://arxiv.org/abs/2204.02854")
*   Code:  [GitHub – Shi-Yupeng/RESAIL-For-SIS: Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis (CVPR2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Shi-Yupeng/RESAIL-For-SIS "GitHub - Shi-Yupeng/RESAIL-For-SIS: Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis (CVPR2022)")

**A Closer Look at Few-shot Image Generation**

*   Paper:  [https://arxiv.org/abs/2205.03805](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.03805 "https://arxiv.org/abs/2205.03805")
*   Tags: Few-shot

**Ensembling Off-the-shelf Models for GAN Training**

*   Paper:  [https://arxiv.org/abs/2112.09130](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.09130 "https://arxiv.org/abs/2112.09130")
*   Code:  [https://github.com/nupurkmr9/vision-aided-gan](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/nupurkmr9/vision-aided-gan "https://github.com/nupurkmr9/vision-aided-gan")

**Few-Shot Font Generation by Learning Fine-Grained Local Styles**

*   Paper:  [https://arxiv.org/abs/2205.09965](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.09965 "https://arxiv.org/abs/2205.09965")
*   Tags: Few-shot

**Modeling Image Composition for Complex Scene Generation**

*   Paper:  [https://arxiv.org/abs/2206.00923](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.00923 "https://arxiv.org/abs/2206.00923")
*   Code:  [GitHub – JohnDreamer/TwFA](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JohnDreamer/TwFA "GitHub - JohnDreamer/TwFA")

**Global Context With Discrete Diffusion in Vector Quantized Modeling for Image Generation**

*   Paper:  [https://arxiv.org/abs/2112.01799](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.01799 "https://arxiv.org/abs/2112.01799")

**Self-supervised Correlation Mining Network for Person Image Generation**

*   Paper:  [https://arxiv.org/abs/2111.13307](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.13307 "https://arxiv.org/abs/2111.13307")

**Learning To Memorize Feature Hallucination for One-Shot Image Generation**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Xie_Learning_To_Memorize_Feature_Hallucination_for_One-Shot_Image_Generation_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Local Attention Pyramid for Scene Image Generation**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Shim_Local_Attention_Pyramid_for_Scene_Image_Generation_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**High-Resolution Image Synthesis with Latent Diffusion Models**

*   Paper:  [https://arxiv.org/abs/2112.10752](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.10752 "https://arxiv.org/abs/2112.10752")
*   Code:  [GitHub – CompVis/latent-diffusion: High-Resolution Image Synthesis with Latent Diffusion Models](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/CompVis/latent-diffusion "GitHub - CompVis/latent-diffusion: High-Resolution Image Synthesis with Latent Diffusion Models")

**Cluster-guided Image Synthesis with Unconditional Models**

*   Paper:  [https://arxiv.org/abs/2112.12911](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.12911 "https://arxiv.org/abs/2112.12911")

**SphericGAN: Semi-Supervised Hyper-Spherical Generative Adversarial Networks for Fine-Grained Image Synthesis**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Chen_SphericGAN_Semi-Supervised_Hyper-Spherical_Generative_Adversarial_Networks_for_Fine-Grained_Image_Synthesis_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**DPGEN: Differentially Private Generative Energy-Guided Network for Natural Image Synthesis**

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Chen_DPGEN_Differentially_Private_Generative_Energy-Guided_Network_for_Natural_Image_Synthesis_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**DO-GAN: A Double Oracle Framework for Generative Adversarial Networks**

*   Paper:  [https://openaccess.thecvf.com/content/CVPR2022/html/Aung\_DO-GAN\_A\_Double\_Oracle\_Framework\_for\_Generative\_Adversarial\_Networks\_CVPR\_2022\_paper.html](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Aung_DO-GAN_A_Double_Oracle_Framework_for_Generative_Adversarial_Networks_CVPR_2022_paper.html "https://openaccess.thecvf.com/content/CVPR2022/html/Aung_DO-GAN_A_Double_Oracle_Framework_for_Generative_Adversarial_Networks_CVPR_2022_paper.html")

**Improving GAN Equilibrium by Raising Spatial Awareness**

*   Paper:  [https://arxiv.org/abs/2112.00718](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.00718 "https://arxiv.org/abs/2112.00718")
*   Code:  [https://github.com/genforce/eqgan-sa](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/genforce/eqgan-sa "https://github.com/genforce/eqgan-sa")

\*\*Polymorphic-GAN: Generating Aligned Samples Across Multiple Domains With Learned Morph Maps\*\*

*   Paper:  [CVPR 2022 Open Access Repository](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Polymorphic-GAN_Generating_Aligned_Samples_Across_Multiple_Domains_With_Learned_Morph_CVPR_2022_paper.html "CVPR 2022 Open Access Repository")

**Manifold Learning Benefits GANs**

*   Paper:  [https://arxiv.org/abs/2112.12618](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.12618 "https://arxiv.org/abs/2112.12618")

**Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data**

*   Paper:  [https://arxiv.org/abs/2204.04950](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.04950 "https://arxiv.org/abs/2204.04950")
*   Code:  [GitHub – FriedRonaldo/Primitives-PS: Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data – Official PyTorch Implementation (CVPR 2022)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/FriedRonaldo/Primitives-PS "GitHub - FriedRonaldo/Primitives-PS: Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data - Official PyTorch Implementation (CVPR 2022)")

**On Conditioning the Input Noise for Controlled Image Generation with Diffusion Models**

*   Paper:  [https://arxiv.org/abs/2205.03859](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.03859 "https://arxiv.org/abs/2205.03859")
*   Tags: \[Workshop\]

**Generate and Edit Your Own Character in a Canonical View**

*   Paper:  [https://arxiv.org/abs/2205.02974](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.02974 "https://arxiv.org/abs/2205.02974")
*   Tags: \[Workshop\]

**StyLandGAN: A StyleGAN based Landscape Image Synthesis using Depth-map**

*   Paper:  [https://arxiv.org/abs/2205.06611](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.06611 "https://arxiv.org/abs/2205.06611")
*   Tags: \[Workshop\]

**Overparameterization Improves StyleGAN Inversion**

*   Paper:  [https://arxiv.org/abs/2205.06304](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.06304 "https://arxiv.org/abs/2205.06304")
*   Tags: \[Workshop\]

### Video Generation/Synthesis

**Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning**

*   Paper:  [https://arxiv.org/abs/2203.02573](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.02573 "https://arxiv.org/abs/2203.02573")
*   Code:  [https://github.com/snap-research/MMVID](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/snap-research/MMVID "https://github.com/snap-research/MMVID")

**Playable Environments: Video Manipulation in Space and Time**

*   Paper:  [https://arxiv.org/abs/2203.01914](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.01914 "https://arxiv.org/abs/2203.01914")
*   Code:  [https://github.com/willi-menapace/PlayableEnvironments](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/willi-menapace/PlayableEnvironments "https://github.com/willi-menapace/PlayableEnvironments")

**StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2**

*   Paper:  [https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf "https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf")
*   Code:  [https://github.com/universome/stylegan-v](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/universome/stylegan-v "https://github.com/universome/stylegan-v")

**Thin-Plate Spline Motion Model for Image Animation**

*   Paper:  [https://arxiv.org/abs/2203.14367](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.14367 "https://arxiv.org/abs/2203.14367")
*   Code:  [GitHub – yoyo-nb/Thin-Plate-Spline-Motion-Model: \[CVPR 2022\] Thin-Plate Spline Motion Model for Image Animation.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model "GitHub - yoyo-nb/Thin-Plate-Spline-Motion-Model: [CVPR 2022] Thin-Plate Spline Motion Model for Image Animation.")

**Make It Move: Controllable Image-to-Video Generation with Text Descriptions**

*   Paper:  [https://arxiv.org/abs/2112.02815](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.02815 "https://arxiv.org/abs/2112.02815")
*   Code:  [GitHub – Youncy-Hu/MAGE: Make It Move: Controllable Image-to-Video Generation with Text Descriptions](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Youncy-Hu/MAGE "GitHub - Youncy-Hu/MAGE: Make It Move: Controllable Image-to-Video Generation with Text Descriptions")

**Diverse Video Generation from a Single Video**

*   Paper:  [https://arxiv.org/abs/2205.05725](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.05725 "https://arxiv.org/abs/2205.05725")
*   Tags: \[Workshop\]

Others
------

**GAN-Supervised Dense Visual Alignment**

*   Paper:  [https://arxiv.org/abs/2112.05143](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.05143 "https://arxiv.org/abs/2112.05143")
*   Code:  [https://github.com/wpeebles/gangealing](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wpeebles/gangealing "https://github.com/wpeebles/gangealing")

**ClothFormer: Taming Video Virtual Try-on in All Module**

*   Paper:  [https://arxiv.org/abs/2204.12151](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.12151 "https://arxiv.org/abs/2204.12151")
*   Tags: Video Virtual Try-on

**Iterative Deep Homography Estimation**

*   Paper:  [https://arxiv.org/abs/2203.15982](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.15982 "https://arxiv.org/abs/2203.15982")
*   Code:  [GitHub – imdumpl78/IHN: This is the open source implementation of the CVPR2022 paper “Iterative Deep Homography Estimation”](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/imdumpl78/IHN "GitHub - imdumpl78/IHN: This is the open source implementation of the CVPR2022 paper")

**Style-Structure Disentangled Features and Normalizing Flows for Diverse Icon Colorization**

*   Paper:  [https://openaccess.thecvf.com/content/CVPR2022/papers/Li\_Style-Structure\_Disentangled\_Features\_and\_Normalizing\_Flows\_for\_Diverse\_Icon\_Colorization\_CVPR\_2022\_paper.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Style-Structure_Disentangled_Features_and_Normalizing_Flows_for_Diverse_Icon_Colorization_CVPR_2022_paper.pdf "https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Style-Structure_Disentangled_Features_and_Normalizing_Flows_for_Diverse_Icon_Colorization_CVPR_2022_paper.pdf")
*   Code:  [GitHub – djosix/IconFlow: Code for “Style-Structure Disentangled Features and Normalizing Flows for Diverse Icon Colorization”, CVPR 2022.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/djosix/IconFlow "GitHub - djosix/IconFlow: Code for")

**Unsupervised Homography Estimation with Coplanarity-Aware GAN**

*   Paper:  [https://arxiv.org/abs/2205.03821](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.03821 "https://arxiv.org/abs/2205.03821")
*   Code:  [GitHub – megvii-research/HomoGAN: This is the official implementation of HomoGAN, CVPR2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/megvii-research/HomoGAN "GitHub - megvii-research/HomoGAN: This is the official implementation of HomoGAN, CVPR2022")

**Diverse Image Outpainting via GAN Inversion**

*   Paper:  [https://arxiv.org/abs/2104.00675](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2104.00675 "https://arxiv.org/abs/2104.00675")
*   Code:  [GitHub – yccyenchicheng/InOut: Diverse Image Outpainting via GAN Inversion](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yccyenchicheng/InOut "GitHub - yccyenchicheng/InOut: Diverse Image Outpainting via GAN Inversion")

**On Aliased Resizing and Surprising Subtleties in GAN Evaluation**

*   Paper:  [https://arxiv.org/abs/2104.11222](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2104.11222 "https://arxiv.org/abs/2104.11222")
*   Code:  [GitHub – GaParmar/clean-fid: PyTorch – FID calculation with proper image resizing and quantization steps \[CVPR 2022\]](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/GaParmar/clean-fid "GitHub - GaParmar/clean-fid: PyTorch - FID calculation with proper image resizing and quantization steps [CVPR 2022]")

**Patch-wise Contrastive Style Learning for Instagram Filter Removal**

*   Paper:  [https://arxiv.org/abs/2204.07486](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.07486 "https://arxiv.org/abs/2204.07486")
*   Code:  [GitHub – birdortyedi/cifr-pytorch](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/birdortyedi/cifr-pytorch "GitHub - birdortyedi/cifr-pytorch")
*   Tags: \[Workshop\]

NTIRE2022
---------

New Trends in Image Restoration and Enhancement workshop and challenges on image and video processing.

### Spectral Reconstruction from RGB

**MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction**

*   Paper:  [https://arxiv.org/abs/2204.07908](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.07908 "https://arxiv.org/abs/2204.07908")
*   Code:  [GitHub – caiyuanhao1998/MST-plus-plus: “MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction” (CVPRW 2022) & (Winner of NTIRE 2022 Spectral Recovery Challenge) and a toolbox for spectral reconstruction](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/caiyuanhao1998/MST-plus-plus "GitHub - caiyuanhao1998/MST-plus-plus:")
*   Tags: 1st place

### Perceptual Image Quality Assessment: Track 1 Full-Reference / Track 2 No-Reference

**MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment**

*   Paper:  [https://arxiv.org/abs/2204.08958](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.08958 "https://arxiv.org/abs/2204.08958")
*   Code:  [GitHub – IIGROUP/MANIQA: \[CVPRW 2022\] MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/IIGROUP/MANIQA "GitHub - IIGROUP/MANIQA: [CVPRW 2022] MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment")
*   Tags: 1st place for track2

**Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network**

*   Paper:  [https://arxiv.org/abs/2204.10485](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.10485 "https://arxiv.org/abs/2204.10485")
*   Code:  [GitHub – IIGROUP/AHIQ: \[CVPRW 2022\] Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/IIGROUP/AHIQ "GitHub - IIGROUP/AHIQ: [CVPRW 2022] Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network")
*   Tags: 1st place for track1

**MSTRIQ: No Reference Image Quality Assessment Based on Swin Transformer with Multi-Stage Fusion**

*   Paper:  [https://arxiv.org/abs/2205.10101](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.10101 "https://arxiv.org/abs/2205.10101")
*   Tags: 2nd place in track2

**Conformer and Blind Noisy Students for Improved Image Quality Assessment**

*   Paper:  [https://arxiv.org/abs/2204.12819](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.12819 "https://arxiv.org/abs/2204.12819")

### Inpainting: Track 1 Unsupervised / Track 2 Semantic

**GLaMa: Joint Spatial and Frequency Loss for General Image Inpainting**

*   Paper:  [https://arxiv.org/abs/2205.07162](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.07162 "https://arxiv.org/abs/2205.07162")
*   Tags: ranked first in terms of PSNR, LPIPS and SSIM in the track1

### Efficient Super-Resolution

*   **Report** :  [https://arxiv.org/abs/2205.05675](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.05675 "https://arxiv.org/abs/2205.05675")

**ShuffleMixer: An Efficient ConvNet for Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2205.15175](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.15175 "https://arxiv.org/abs/2205.15175")
*   Code:  [https://github.com/sunny2109/MobileSR-NTIRE2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sunny2109/MobileSR-NTIRE2022 "https://github.com/sunny2109/MobileSR-NTIRE2022")
*   Tags: Winner of the model complexity track

**Edge-enhanced Feature Distillation Network for Efficient Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2204.08759](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.08759 "https://arxiv.org/abs/2204.08759")
*   Code:  [https://github.com/icandle/EFDN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/icandle/EFDN "https://github.com/icandle/EFDN")

**Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2204.08759](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.08759 "https://arxiv.org/abs/2204.08759")
*   Code:  [GitHub – NJU-Jet/FMEN: Lowest memory consumption and second shortest runtime in NTIRE 2022 challenge on Efficient Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/NJU-Jet/FMEN "GitHub - NJU-Jet/FMEN: Lowest memory consumption and second shortest runtime in NTIRE 2022 challenge on Efficient Super-Resolution")
*   Tags: Lowest memory consumption and second shortest runtime

**Blueprint Separable Residual Network for Efficient Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2205.05996](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.05996 "https://arxiv.org/abs/2205.05996")
*   Code:  [GitHub – xiaom233/BSRN: Blueprint Separable Residual Network for Efficient Image Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/xiaom233/BSRN "GitHub - xiaom233/BSRN: Blueprint Separable Residual Network for Efficient Image Super-Resolution")
*   Tags: 1st place in model complexity track

### Night Photography Rendering

**Rendering Nighttime Image Via Cascaded Color and Brightness Compensation**

*   Paper:  [https://arxiv.org/abs/2204.08970](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.08970 "https://arxiv.org/abs/2204.08970")
*   Code:  [GitHub – NJUVISION/CBUnet: Official code of the “Rendering Nighttime Image Via Cascaded Color and Brightness Compensation”](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/NJUVISION/CBUnet "GitHub - NJUVISION/CBUnet: Official code of the")
*   Tags: 2nd place

### Super-Resolution and Quality Enhancement of Compressed Video: Track1 (Quality enhancement) / Track2 (Quality enhancement and x2 SR) / Track3 (Quality enhancement and x4 SR)

*   **Report** :  [https://arxiv.org/abs/2204.09314](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.09314 "https://arxiv.org/abs/2204.09314")
*   **Homepage** :  [GitHub – RenYang-home/NTIRE22\_VEnh\_SR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/RenYang-home/NTIRE22_VEnh_SR "GitHub - RenYang-home/NTIRE22_VEnh_SR")

**Progressive Training of A Two-Stage Framework for Video Restoration**

*   Paper:  [https://arxiv.org/abs/2204.09924](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.09924 "https://arxiv.org/abs/2204.09924")
*   Code:  [GitHub – ryanxingql/winner-ntire22-vqe: Our method and experience of winning the NTIRE22 challenge on video quality enhancement](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ryanxingql/winner-ntire22-vqe "GitHub - ryanxingql/winner-ntire22-vqe: Our method and experience of winning the NTIRE22 challenge on video quality enhancement")
*   Tags: 1st place in track1 and track2, 2nd place in track3

### High Dynamic Range (HDR): Track 1 Low-complexity (fidelity constraint) / Track 2 Fidelity (low-complexity constraint)

*   **Report** :  [https://arxiv.org/abs/2205.12633](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.12633 "https://arxiv.org/abs/2205.12633")

**Efficient Progressive High Dynamic Range Image Restoration via Attention and Alignment Network**

*   Paper:  [https://arxiv.org/abs/2204.09213](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.09213 "https://arxiv.org/abs/2204.09213")
*   Tags: 2nd palce of both two tracks

### Stereo Super-Resolution

*   **Report** :  [https://arxiv.org/abs/2204.09197](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.09197 "https://arxiv.org/abs/2204.09197")

**Parallel Interactive Transformer**

*   Code:  [GitHub – chaineypung/CVPR-NTIRE2022-Parallel-Interactive-Transformer: This is the source code of the 7th place solution for stereo image super resolution task in 2022 CVPR NTIRE challenge.](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/chaineypung/CVPR-NTIRE2022-Parallel-Interactive-Transformer-PAIT "GitHub - chainypung/CVPR-NTIRE2022-Parallel-Interactive-Transformer: This is the source code of the 7th place solution for stereo image super resolution task in 2022 CVPR NTIRE challenge.")
*   Tags: 7st place

### Burst Super-Resolution: Track 2 Real

**BSRT: Improving Burst Super-Resolution with Swin Transformer and Flow-Guided Deformable Alignment**

*   Code:  [https://github.com/Algolzw/BSRT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Algolzw/BSRT "https://github.com/Algolzw/BSRT")
*   Tags: 1st place

ECCV2022-Low-Level-Vision
=========================

Image Restoration – Image Restoration
-------------------------------------

**Simple Baselines for Image Restoration**

*   Paper:  [https://arxiv.org/abs/2204.04676](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.04676 "https://arxiv.org/abs/2204.04676")
*   Code:  [https://github.com/megvii-research/NAFNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/megvii-research/NAFNet "https://github.com/megvii-research/NAFNet")

**D2HNet: Joint Denoising and Deblurring with Hierarchical Network for Robust Night Image Restoration**

*   Paper:  [https://arxiv.org/abs/2207.03294](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.03294 "https://arxiv.org/abs/2207.03294")
*   Code:  [https://github.com/zhaoyuzhi/D2HNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zhaoyuzhi/D2HNet "https://github.com/zhaoyuzhi/D2HNet")

**Seeing Far in the Dark with Patterned Flash**

*   Paper:  [https://arxiv.org/abs/2207.12570](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.12570 "https://arxiv.org/abs/2207.12570")
*   Code:  [https://github.com/zhsun0357/Seeing-Far-in-the-Dark-with-Patterned-Flash](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zhsun0357/Seeing-Far-in-the-Dark-with-Patterned-Flash "https://github.com/zhsun0357/Seeing-Far-in-the-Dark-with-Patterned-Flash")

**BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks**

*   Paper:  [https://arxiv.org/abs/2207.06873](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.06873 "https://arxiv.org/abs/2207.06873")
*   Code:  [https://github.com/ExplainableML/BayesCap](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ExplainableML/BayesCap "https://github.com/ExplainableML/BayesCap")

**Improving Image Restoration by Revisiting Global Information Aggregation**

*   Paper:  [https://arxiv.org/abs/2112.04491](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.04491 "https://arxiv.org/abs/2112.04491")
*   Code:  [https://github.com/megvii-research/TLC](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/megvii-research/TLC "https://github.com/megvii-research/TLC")

**Fast Two-step Blind Optical Aberration Correction**

*   Paper:  [https://arxiv.org/abs/2208.00950](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.00950 "https://arxiv.org/abs/2208.00950")
*   Code:  [https://github.com/teboli/fast\_two\_stage\_psf\_correction](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/teboli/fast_two_stage_psf_correction "https://github.com/teboli/fast_two_stage_psf_correction")
*   Tags: Optical Aberration Correction

**VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder**

*   Paper:  [https://arxiv.org/abs/2205.06803](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.06803 "https://arxiv.org/abs/2205.06803")
*   Code:  [https://github.com/TencentARC/VQFR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/TencentARC/VQFR "https://github.com/TencentARC/VQFR")
*   Tags: Blind Face Restoration

**RAWtoBit: A Fully End-to-end Camera ISP Network**

*   Paper:  [https://arxiv.org/abs/2208.07639](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.07639 "https://arxiv.org/abs/2208.07639")
*   Tags: ISP and Image Compression

**Transform your Smartphone into a DSLR Camera: Learning the ISP in the Wild**

*   Paper:  [https://arxiv.org/abs/2203.10636](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.10636 "https://arxiv.org/abs/2203.10636")
*   Code:  [https://github.com/4rdhendu/TransformPhone2DSLR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/4rdhendu/TransformPhone2DSLR "https://github.com/4rdhendu/TransformPhone2DSLR")
*   Tags: ISP

**Single Frame Atmospheric Turbulence Mitigation: A Benchmark Study and A New Physics-Inspired Transformer Model**

*   Paper:  [https://arxiv.org/abs/2207.10040](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10040 "https://arxiv.org/abs/2207.10040")
*   Code:  [https://github.com/VITA-Group/TurbNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/VITA-Group/TurbNet "https://github.com/VITA-Group/TurbNet")
*   Tags: Atmospheric Turbulence Mitigation, Transformer

**Modeling Mask Uncertainty in Hyperspectral Image Reconstruction**

*   Paper:  [https://arxiv.org/abs/2112.15362](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.15362 "https://arxiv.org/abs/2112.15362")
*   Code:  [https://github.com/Jiamian-Wang/mask\_uncertainty\_spectral\_SCI](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Jiamian-Wang/mask_uncertainty_spectral_SCI "https://github.com/Jiamian-Wang/mask_uncertainty_spectral_SCI")
*   Tags: Hyperspectral Image Reconstruction

**TAPE: Task-Agnostic Prior Embedding for Image Restoration**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3292_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**DRCNet: Dynamic Image Restoration Contrastive Network**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6389_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**ART-SS: An Adaptive Rejection Technique for Semi-Supervised Restoration for Adverse Weather-Affected Images**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4237_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/rajeevyasarla/ART-SS](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/rajeevyasarla/ART-SS "https://github.com/rajeevyasarla/ART-SS")
*   Tags: Adverse Weather

**Spectrum-Aware and Transferable Architecture Search for Hyperspectral Image Restoration**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4367_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Hyperspectral Image Restoration

**Seeing through a Black Box: Toward High-Quality Terahertz Imaging via Subspace-and-Attention Guided Restoration**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6259_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Terahertz Imaging

**JPEG Artifacts Removal via Contrastive Representation Learning**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/171_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: JPEG Artifacts Removal

**Zero-Shot Learning for Reflection Removal of Single 360-Degree Image**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6418_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Reflection Removal

**Overexposure Mask Fusion: Generalizable Reverse ISP Multi-Step Refinement**

*   Paper:  [https://arxiv.org/abs/2210.11511](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2210.11511 "https://arxiv.org/abs/2210.11511")
*   Code:  [https://github.com/SenseBrainTech/overexposure-mask-reverse-ISP](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/SenseBrainTech/overexposure-mask-reverse-ISP "https://github.com/SenseBrainTech/overexposure-mask-reverse-ISP")
*   Tagss: \[Workshop\], Reversed ISP

### Video Restoration

**Video Restoration Framework and Its Meta-Adaptations to Data-Poor Conditions**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7533_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

Super Resolution – super resolution
-----------------------------------

### Image Super Resolution

**ARM: Any-Time Super-Resolution Method**

*   Paper:  [https://arxiv.org/abs/2203.10812](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.10812 "https://arxiv.org/abs/2203.10812")
*   Code:  [https://github.com/chenbong/ARM-Net](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/chenbong/ARM-Net "https://github.com/chenbong/ARM-Net")

**Dynamic Dual Trainable Bounds for Ultra-low Precision Super-Resolution Networks**

*   Paper:  [https://arxiv.org/abs/2203.03844](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.03844 "https://arxiv.org/abs/2203.03844")
*   Code:  [https://github.com/zysxmu/DDTB](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zysxmu/DDTB "https://github.com/zysxmu/DDTB")

**CADyQ : Contents-Aware Dynamic Quantization for Image Super Resolution**

*   Paper:  [https://arxiv.org/abs/2207.10345](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10345 "https://arxiv.org/abs/2207.10345")
*   Code:  [https://github.com/Cheeun/CADyQ](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Cheeun/CADyQ "https://github.com/Cheeun/CADyQ")

**Image Super-Resolution with Deep Dictionary**

*   Paper:  [https://arxiv.org/abs/2207.09228](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.09228 "https://arxiv.org/abs/2207.09228")
*   Code:  [https://github.com/shuntama/srdd](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/shuntama/srdd "https://github.com/shuntama/srdd")

**Perception-Distortion Balanced ADMM Optimization for Single-Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2208.03324](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.03324 "https://arxiv.org/abs/2208.03324")
*   Code:  [https://github.com/Yuehan717/PDASR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Yuehan717/PDASR "https://github.com/Yuehan717/PDASR")

**Adaptive Patch Exiting for Scalable Single Image Super-Resolution**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2021_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/littlepure2333/APE](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/littlepure2333/APE "https://github.com/littlepure2333/APE")

**Learning Series-Parallel Lookup Tables for Efficient Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2207.12987](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.12987 "https://arxiv.org/abs/2207.12987")
*   Code:  [https://github.com/zhjy2016/SPLUT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zhjy2016/SPLUT "https://github.com/zhjy2016/SPLUT")
*   Tags: Efficient

**MuLUT: Cooperating Mulitple Look-Up Tables for Efficient Image Super-Resolution**

*   Paper:  [https://www.ecva.net/papers/eccv\_2022/papers\_ECCV/papers/136780234.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780234.pdf "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780234.pdf")
*   Code:  [https://github.com/ddlee-cn/MuLUT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ddlee-cn/MuLUT "https://github.com/ddlee-cn/MuLUT")
*   Tags: Efficient

**Efficient Long-Range Attention Network for Image Super-resolution**

*   Paper:  [https://arxiv.org/abs/2203.06697](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.06697 "https://arxiv.org/abs/2203.06697")
*   Code:  [https://github.com/xindongzhang/ELAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/xindongzhang/ELAN "https://github.com/xindongzhang/ELAN")

**Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2207.12577](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.12577 "https://arxiv.org/abs/2207.12577")
*   Code:  [https://github.com/wuyushuwys/compiler-aware-nas-sr](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wuyushuwys/compiler-aware-nas-sr "https://github.com/wuyushuwys/compiler-aware-nas-sr")

**Restore Globally, Refine Locally: A Mask-Guided Scheme to Accelerate Super-Resolution Networks**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4417_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/huxiaotaostasy/MGA-scheme](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/huxiaotaostasy/MGA-scheme "https://github.com/huxiaotaostasy/MGA-scheme")

**Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2207.09156](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.09156 "https://arxiv.org/abs/2207.09156")
*   Code:  [https://github.com/palmdong/MMSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/palmdong/MMSR "https://github.com/palmdong/MMSR")
*   Tags: Self-Supervised

**Self-Supervised Learning for Real-World Super-Resolution from Dual Zoomed Observations**

*   Paper:  [https://arxiv.org/abs/2203.01325](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.01325 "https://arxiv.org/abs/2203.01325")
*   Code:  [https://github.com/cszhilu1998/SelfDZSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/cszhilu1998/SelfDZSR "https://github.com/cszhilu1998/SelfDZSR")
*   Tags: Self-Supervised, Reference-based

**Efficient and Degradation-Adaptive Network for Real-World Image Super-Resolution**

*   Paper:  [http://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022\_DASR.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=http://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022_DASR.pdf "http://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022_DASR.pdf")
*   Code:  [https://github.com/csjliang/DASR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/csjliang/DASR "https://github.com/csjliang/DASR")
*   Tags: Real World

**D2C-SR: A Divergence to Convergence Approach for Real-World Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2103.14373](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2103.14373 "https://arxiv.org/abs/2103.14373")
*   Code:  [https://github.com/megvii-research/D2C-SR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/megvii-research/D2C-SR "https://github.com/megvii-research/D2C-SR")
*   Tag: Real World

**MM-RealSR: Metric Learning based Interactive Modulation for Real-World Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2205.05065](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.05065 "https://arxiv.org/abs/2205.05065")
*   Code:  [https://github.com/TencentARC/MM-RealSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/TencentARC/MM-RealSR "https://github.com/TencentARC/MM-RealSR")
*   Tag: Real World

**KXNet: A Model-Driven Deep Neural Network for Blind Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2209.10305](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.10305 "https://arxiv.org/abs/2209.10305")
*   Code:  [https://github.com/jiahong-fu/KXNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jiahong-fu/KXNet "https://github.com/jiahong-fu/KXNet")
*   Tags: Blind

**From Face to Natural Image: Learning Real Degradation for Blind Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2210.00752](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2210.00752 "https://arxiv.org/abs/2210.00752")
*   Code:  [https://github.com/csxmli2016/ReDegNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/csxmli2016/ReDegNet "https://github.com/csxmli2016/ReDegNet")
*   Tags: Blind

**Unfolded Deep Kernel Estimation for Blind Image Super-Resolution**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3484_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/natezhenghy/UDKE](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/natezhenghy/UDKE "https://github.com/natezhenghy/UDKE")
*   Tags: Blind

**Uncertainty Learning in Kernel Estimation for Multi-stage Blind Image Super-Resolution**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1649_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Blind

**Super-Resolution by Predicting Offsets: An Ultra-Efficient Super-Resolution Network for Rasterized Images**

*   Paper:  [https://arxiv.org/abs/2210.04198](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2210.04198 "https://arxiv.org/abs/2210.04198")
*   Code:  [https://github.com/HaomingCai/SRPO](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/HaomingCai/SRPO "https://github.com/HaomingCai/SRPO")
*   Tags: Rasterized Images

**Reference-based Image Super-Resolution with Deformable Attention Transformer**

*   Paper:  [https://arxiv.org/abs/2207.11938](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.11938 "https://arxiv.org/abs/2207.11938")
*   Code:  [https://github.com/caojiezhang/DATSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/caojiezhang/DATSR "https://github.com/caojiezhang/DATSR")
*   Tags: Reference-based, Transformer

**RRSR: Reciprocal Reference-Based Image Super-Resolution with Progressive Feature Alignment and Selection**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7808_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Reference-based

**Boosting Event Stream Super-Resolution with a Recurrent Neural Network**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/248_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Event

**HST: Hierarchical Swin Transformer for Compressed Image Super-resolution**

*   Paper:  [https://arxiv.org/abs/2208.09885](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.09885 "https://arxiv.org/abs/2208.09885")
*   Tags: \[Workshop-AIM2022\]

**Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration**

*   Paper:  [https://arxiv.org/abs/2209.11345](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.11345 "https://arxiv.org/abs/2209.11345")
*   Code:  [https://github.com/mv-lab/swin2sr](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/mv-lab/swin2sr "https://github.com/mv-lab/swin2sr")
*   Tags: \[Workshop-AIM2022\]

**Fast Nearest Convolution for Real-Time Efficient Image Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2208.11609](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.11609 "https://arxiv.org/abs/2208.11609")
*   Code:  [https://github.com/Algolzw/NCNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Algolzw/NCNet "https://github.com/Algolzw/NCNet")
*   Tags: \[Workshop-AIM2022\]

### Video Super Resolution

**Learning Spatiotemporal Frequency-Transformer for Compressed Video Super-Resolution**

*   Paper:  [https://arxiv.org/abs/2208.03012](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.03012 "https://arxiv.org/abs/2208.03012")
*   Code:  [https://github.com/researchmm/FTVSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/researchmm/FTVSR "https://github.com/researchmm/FTVSR")
*   Tags: Compressed Video SR

**A Codec Information Assisted Framework for Efficient Compressed Video Super-Resolution**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6420_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Compressed Video SR

**Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset**

*   Paper:  [https://arxiv.org/abs/2209.12475](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.12475 "https://arxiv.org/abs/2209.12475")
*   Code:  [https://github.com/zmzhang1998/Real-RawVSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zmzhang1998/Real-RawVSR "https://github.com/zmzhang1998/Real-RawVSR")

Denoising – denoising
---------------------

### Image Denoising

**Deep Semantic Statistics Matching (D2SM) Denoising Network**

*   Paper:  [https://arxiv.org/abs/2207.09302](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.09302 "https://arxiv.org/abs/2207.09302")
*   Code:  [https://github.com/MKFMIKU/d2sm](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/MKFMIKU/d2sm "https://github.com/MKFMIKU/d2sm")

**Fast and High Quality Image Denoising via Malleable Convolution**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3257_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

### Video Denoising

**Unidirectional Video Denoising by Mimicking Backward Recurrent Modules with Look-ahead Forward Ones**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4024_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/nagejacob/FloRNN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/nagejacob/FloRNN "https://github.com/nagejacob/FloRNN")

**TempFormer: Temporally Consistent Transformer for Video Denoising**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6092_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Transformer

Deblurring – Deblurring
-----------------------

### Image Deblurring

**Learning Degradation Representations for Image Deblurring**

*   Paper:  [https://arxiv.org/abs/2208.05244](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.05244 "https://arxiv.org/abs/2208.05244")
*   Code:  [https://github.com/dasongli1/Learning\_degradation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/dasongli1/Learning_degradation "https://github.com/dasongli1/Learning_degradation")

**Stripformer: Strip Transformer for Fast Image Deblurring**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4651_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Transformer

**Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance**

*   Paper:  [https://arxiv.org/abs/2207.10123](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10123 "https://arxiv.org/abs/2207.10123")
*   Code:  [https://github.com/zzh-tech/Animation-from-Blur](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zzh-tech/Animation-from-Blur "https://github.com/zzh-tech/Animation-from-Blur")
*   Tags: recovering detailed motion from a single motion-blurred image

**United Defocus Blur Detection and Deblurring via Adversarial Promotion Learning**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3308_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/wdzhao123/APL](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wdzhao123/APL "https://github.com/wdzhao123/APL")
*   Tags: Defocus Blur

**Realistic Blur Synthesis for Learning Image Deblurring**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6325_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Blur Synthesis

**Event-based Fusion for Motion Deblurring with Cross-modal Attention**

*   Paper: [https://arxiv.org/abs/2112.00167](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.00167 "https://arxiv.org/abs/2112.00167")
*   Code:  [https://github.com/AHupuJR/EFNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/AHupuJR/EFNet "https://github.com/AHupuJR/EFNet")
*   Tags: Event-based

**Event-Guided Deblurring of Unknown Exposure Time Videos**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3601_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Event-based

### Video Deblurring

**Spatio-Temporal Deformable Attention Network for Video Deblurring**

*   Paper:  [https://arxiv.org/abs/2207.10852](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10852 "https://arxiv.org/abs/2207.10852")
*   Code:  [https://github.com/huicongzhang/STDAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/huicongzhang/STDAN "https://github.com/huicongzhang/STDAN")

**Efficient Video Deblurring Guided by Motion Magnitude**

*   Paper:  [https://arxiv.org/abs/2207.13374](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.13374 "https://arxiv.org/abs/2207.13374")
*   Code:  [https://github.com/sollynoay/MMP-RNN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sollynoay/MMP-RNN "https://github.com/sollynoay/MMP-RNN")

**ERDN: Equivalent Receptive Field Deformable Network for Video Deblurring**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4085_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/TencentCloud/ERDN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/TencentCloud/ERDN "https://github.com/TencentCloud/ERDN")

**DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting**

*   Paper:  [https://arxiv.org/abs/2111.09985](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.09985 "https://arxiv.org/abs/2111.09985")
*   Code:  [https://github.com/JihyongOh/DeMFI](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JihyongOh/DeMFI "https://github.com/JihyangOh/DeMFI")
*   Tags: Joint Deblurring and Frame Interpolation

**Towards Real-World Video Deblurring by Exploring Blur Formation Process**

*   Paper:  [https://arxiv.org/abs/2208.13184](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.13184 "https://arxiv.org/abs/2208.13184")
*   Tags: \[Workshop-AIM2022\]

Image Decomposition
-------------------

**Blind Image Decomposition**

*   Paper:  [https://arxiv.org/abs/2108.11364](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2108.11364 "https://arxiv.org/abs/2108.11364")
*   Code:  [https://github.com/JunlinHan/BID](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JunlinHan/BID "https://github.com/JunlinHan/BID")

Deraining – deraining
---------------------

**Not Just Streaks: Towards Ground Truth for Single Image Deraining**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1506_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/UCLA-VMG/GT-RAIN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/UCLA-VMG/GT-RAIN "https://github.com/UCLA-VMG/GT-RAIN")

**Rethinking Video Rain Streak Removal: A New Synthesis Model and a Deraining Network with Video Rain Prior**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6798_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/wangshauitj/RDD-Net](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wangshauitj/RDD-Net "https://github.com/wangshauitj/RDD-Net")

Dehazing – to fog
-----------------

**Frequency and Spatial Dual Guidance for Image Dehazing**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4734_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/yuhuUSTC/FSDGN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/yuhuUSTC/FSDGN "https://github.com/yuhuUSTC/FSDGN")

**Perceiving and Modeling Density for Image Dehazing**

*   Paper:  [https://arxiv.org/abs/2111.09733](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.09733 "https://arxiv.org/abs/2111.09733")
*   Code:  [https://github.com/Owen718/ECCV22-Perceiving-and-Modeling-Density-for-Image-Dehazing](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Owen718/ECCV22-Perceiving-and-Modeling-Density-for-Image-Dehazing "https://github.com/Owen718/ECCV22-Perceiving-and-Modeling-Density-for-Image-Dehazing")

**Boosting Supervised Dehazing Methods via Bi-Level Patch Reweighting**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1346_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Unpaired Deep Image Dehazing Using Contrastive Disentanglement Learning**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/255_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

Demoireing – Go moiré
---------------------

**Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing**

*   Paper:  [https://arxiv.org/abs/2207.09935](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.09935 "https://arxiv.org/abs/2207.09935")
*   Code:  [https://github.com/XinYu-Andy/uhdm-page](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/XinYu-Andy/uhdm-page "https://github.com/XinYu-Andy/uhdm-page")

HDR Imaging / Multi-Exposure Image Fusion – HDR image generation / multi-exposure image fusion
----------------------------------------------------------------------------------------------

**Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6250_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/viengiaan/EDWL](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/viengiaan/EDWL "https://github.com/viengiaan/EDWL")

**Ghost-free High Dynamic Range Imaging with Context-aware Transformer**

*   Paper:  [https://arxiv.org/abs/2208.05114](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.05114 "https://arxiv.org/abs/2208.05114")
*   Code:  [https://github.com/megvii-research/HDR-Transformer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/megvii-research/HDR-Transformer "https://github.com/megvii-research/HDR-Transformer")

**Selective TransHDR: Transformer-Based Selective HDR Imaging Using Ghost Region Mask**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6670_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields**

*   Paper:  [https://arxiv.org/abs/2208.06787](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.06787 "https://arxiv.org/abs/2208.06787")
*   Code:  [https://github.com/postech-ami/HDR-Plenoxels](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/postech-ami/HDR-Plenoxels "https://github.com/postech-ami/HDR-Plenoxels")

**Towards Real-World HDRTV Reconstruction: A Data Synthesis-Based Approach**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4873_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

Image Fusion
------------

**FusionVAE: A Deep Hierarchical Variational Autoencoder for RGB Image Fusion**

*   Paper:  [https://arxiv.org/abs/2209.11277](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.11277 "https://arxiv.org/abs/2209.11277")

**Recurrent Correction Network for Fast and Efficient Multi-modality Image Fusion**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3864_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/MisakiCoca/ReCoNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/MisakiCoca/ReCoNet "https://github.com/MisakiCoca/ReCoNet")

**Neural Image Representations for Multi-Image Fusion and Layer Separation**

*   Paper:  [https://arxiv.org/abs/2108.01199](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2108.01199 "https://arxiv.org/abs/2108.01199")
*   Code:  [Seonghyeon Nam | Neural Image Representations for Multi-Image Fusion and Layer Separation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://shnnam.github.io/research/nir/ "Seonghyeon Nam | Neural Image Representations for Multi-Image Fusion and Layer Separation")

**Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4260_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/erfect2020/DecompositionForFusion](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/erfect2020/DecompositionForFusion "https://github.com/erfect2020/DecompositionForFusion")

Frame Interpolation – frame insertion
-------------------------------------

**Real-Time Intermediate Flow Estimation for Video Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2011.06294](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2011.06294 "https://arxiv.org/abs/2011.06294")
*   Code:  [https://github.com/hzwer/ECCV2022-RIFE](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/hzwer/ECCV2022-RIFE "https://github.com/hzwer/ECCV2022-RIFE")

**FILM: Frame Interpolation for Large Motion**

*   Paper:  [https://arxiv.org/abs/2202.04901](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2202.04901 "https://arxiv.org/abs/2202.04901")
*   Code:  [https://github.com/google-research/frame-interpolation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/google-research/frame-interpolation "https://github.com/google-research/frame-interpolation")

**Video Interpolation by Event-driven Anisotropic Adjustment of Optical Flow**

*   Paper:  [https://arxiv.org/abs/2208.09127](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.09127 "https://arxiv.org/abs/2208.09127")

**Learning Cross-Video Neural Representations for High-Quality Frame Interpolation**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2565_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Deep Bayesian Video Frame Interpolation**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1287_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/Oceanlib/DBVI](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Oceanlib/DBVI "https://github.com/Oceanlib/DBVI")

**A Perceptual Quality Metric for Video Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2210.01879](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2210.01879 "https://arxiv.org/abs/2210.01879")
*   Code:  [https://github.com/hqqxyy/VFIPS](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/hqqxyy/VFIPS "https://github.com/hqqxyy/VFIPS")

**DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting**

*   Paper:  [https://arxiv.org/abs/2111.09985](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.09985 "https://arxiv.org/abs/2111.09985")
*   Code:  [https://github.com/JihyongOh/DeMFI](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JihyongOh/DeMFI "https://github.com/JihyangOh/DeMFI")
*   Tags: Joint Deblurring and Frame Interpolation

### Spatial-Temporal Video Super-Resolution

**Towards Interpretable Video Super-Resolution via Alternating Optimization**

*   Paper:  [https://arxiv.org/abs/2207.10765](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10765 "https://arxiv.org/abs/2207.10765")
*   Code:  [https://github.com/caojiezhang/DAVSR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/caojiezhang/DAVSR "https://github.com/caojiezhang/DAVSR")

Image Enhancement – ​​image enhancement
---------------------------------------

**Local Color Distributions Prior to Image Enhancement**

*   Paper:  [https://www.cs.cityu.edu.hk/~rynson/papers/eccv22b.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.cs.cityu.edu.hk/~rynson/papers/eccv22b.pdf "https://www.cs.cityu.edu.hk/~rynson/papers/eccv22b.pdf")
*   Code:  [https://github.com/hywang99/LCDPNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/hywang99/LCDPNet "https://github.com/hywang99/LCDPNet")

**SepLUT: Separable Image-adaptive Lookup Tables for Real-time Image Enhancement**

*   Paper:  [https://arxiv.org/abs/2207.08351](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.08351 "https://arxiv.org/abs/2207.08351")

**Neural Color Operators for Sequential Image Retouching**

*   Paper:  [https://arxiv.org/abs/2207.08080](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.08080 "https://arxiv.org/abs/2207.08080")
*   Code:  [https://github.com/amberwangyili/neurop](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/amberwangyili/neurop "https://github.com/amberwangyili/neurop")

**Deep Fourier-Based Exposure Correction Network with Spatial-Frequency Interaction**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4678_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Exposure Correction

**Uncertainty Inspired Underwater Image Enhancement**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3298_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Underwater Image Enhancement

**NEST: Neural Event Stack for Event-Based Image Enhancement**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2730_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Event-Based

### Low-Light Image Enhancement

**LEDNet: Joint Low-light Enhancement and Deblurring in the Dark**

*   Paper:  [https://arxiv.org/abs/2202.03373](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2202.03373 "https://arxiv.org/abs/2202.03373")
*   Code:  [https://github.com/sczhou/LEDNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sczhou/LEDNet "https://github.com/sczhou/LEDNet")

**Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression**

*   Paper:  [https://arxiv.org/abs/2207.10564](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10564 "https://arxiv.org/abs/2207.10564")
*   Code:  [https://github.com/jinyeying/night-enhancement](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jinyeying/night-enhancement "https://github.com/jinyeying/night-enhancement")

Image Harmonization – Image Harmonization
-----------------------------------------

**Harmonizer: Learning to Perform White-Box Image and Video Harmonization**

*   Paper:  [https://arxiv.org/abs/2207.01322](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.01322 "https://arxiv.org/abs/2207.01322")
*   Code:  [https://github.com/ZHKKKe/Harmonizer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ZHKKKe/Harmonizer "https://github.com/ZHKKKe/Harmonizer")

**DCCF: Deep Comprehensive Color Filter Learning Framework for High-Resolution Image Harmonization**

*   Paper:  [https://arxiv.org/abs/2207.04788](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.04788 "https://arxiv.org/abs/2207.04788")
*   Code:  [https://github.com/rockeyben/DCCF](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/rockeyben/DCCF "https://github.com/rockeyben/DCCF")

**Semantic-Guided Multi-Mask Image Harmonization**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3151_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/XuqianRen/Semantic-guided-Multi-mask-Image-Harmonization](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/XuqianRen/Semantic-guided-Multi-mask-Image-Harmonization "https://github.com/XuqianRen/Semantic-guided-Multi-mask-Image-Harmonization")

**Spatial-Separated Curve Rendering Network for Efficient and High-Resolution Image Harmonization**

*   Paper:  [https://arxiv.org/abs/2109.05750](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2109.05750 "https://arxiv.org/abs/2109.05750")
*   Code:  [https://github.com/stefanLeong/S2CRNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/stefanLeong/S2CRNet "https://github.com/stefanLeong/S2CRNet")

Image Completion/Inpainting – image restoration
-----------------------------------------------

**Learning Prior Feature and Attention Enhanced Image Inpainting**

*   Paper:  [https://arxiv.org/abs/2208.01837](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.01837 "https://arxiv.org/abs/2208.01837")
*   Code:  [https://github.com/ewrfcas/MAE-FAR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ewrfcas/MAE-FAR "https://github.com/ewrfcas/MAE-FAR")

**Perceptual Artifacts Localization for Inpainting**

*   Paper:  [https://arxiv.org/abs/2208.03357](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.03357 "https://arxiv.org/abs/2208.03357")
*   Code:  [https://github.com/owenzlz/PAL4Inpaint](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/owenzlz/PAL4Inpaint "https://github.com/owenzlz/PAL4Inpaint")

**High-Fidelity Image Inpainting with GAN Inversion**

*   Paper:  [https://arxiv.org/abs/2208.11850](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.11850 "https://arxiv.org/abs/2208.11850")

**Unbiased Multi-Modality Guidance for Image Inpainting**

*   Paper:  [https://arxiv.org/abs/2208.11844](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.11844 "https://arxiv.org/abs/2208.11844")

**Image Inpainting with Cascaded Modulation GAN and Object-Aware Training**

*   Paper:  [https://arxiv.org/abs/2203.11947](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.11947 "https://arxiv.org/abs/2203.11947")
*   Code:  [https://github.com/htzheng/CM-GAN-Inpainting](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/htzheng/CM-GAN-Inpainting "https://github.com/htzheng/CM-GAN-Inpainting")

**Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5789_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Diverse Image Inpainting with Normalizing Flow**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2814_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Hourglass Attention Network for Image Inpainting**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3369_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Perceptual Artifacts Localization for Inpainting**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2153_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Don't Forget Me: Accurate Background Recovery for Text Removal via Modeling Local-Global Context**

*   Paper:  [https://arxiv.org/abs/2207.10273](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10273 "https://arxiv.org/abs/2207.10273")
*   Code:  [https://github.com/lcy0604/CTRNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lcy0604/CTRNet "https://github.com/lcy0604/CTRNet")
*   Tags: Text Removal

**The Surprisingly Straightforward Scene Text Removal Method with Gated Attention and Region of Interest Generation: A Comprehensive Prominent Model Analysis**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4705_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/naver/garnet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/naver/garnet "https://github.com/naver/garnet")
*   Tags: Text Removal

### Video Inpainting

**Error Compensation Framework for Flow-Guided Video Inpainting**

*   Paper:  [https://arxiv.org/abs/2207.10391](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10391 "https://arxiv.org/abs/2207.10391")

**Flow-Guided Transformer for Video Inpainting**

*   Paper:  [https://arxiv.org/abs/2208.06768](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.06768 "https://arxiv.org/abs/2208.06768")
*   Code:  [https://github.com/hitachinsk/FGT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/hitachinsk/FGT "https://github.com/hitachinsk/FGT")

Image Colorization – image colorization
---------------------------------------

**Eliminating Gradient Conflict in Reference-based Line-art Colorization**

*   Paper:  [https://arxiv.org/abs/2207.06095](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.06095 "https://arxiv.org/abs/2207.06095")
*   Code:  [https://github.com/kunkun0w0/SGA](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/kunkun0w0/SGA "https://github.com/kunkun0w0/SGA")

**Bridging the Domain Gap towards Generalization in Automatic Colorization**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7304_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/Lhyejin/DG-Colorization](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Lhyejin/DG-Colorization "https://github.com/Lhyejin/DG-Colorization")

**CT2: Colorization Transformer via Color Tokens**

*   Paper:  [https://ci.idm.pku.edu.cn/Weng\_ECCV22b.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://ci.idm.pku.edu.cn/Weng_ECCV22b.pdf "https://ci.idm.pku.edu.cn/Weng_ECCV22b.pdf")
*   Code:  [https://github.com/shuchenweng/CT2](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/shuchenweng/CT2 "https://github.com/shuchenweng/CT2")

**PalGAN: Image Colorization with Palette Generative Adversarial Networks**

*   Paper:  [https://arxiv.org/abs/2210.11204](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2210.11204 "https://arxiv.org/abs/2210.11204")
*   Code:  [https://github.com/shepnerd/PalGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/shepnerd/PalGAN "https://github.com/shepnerd/PalGAN")

**BigColor: Colorization using a Generative Color Prior for Natural Images**

*   Paper:  [https://kimgeonung.github.io/assets/bigcolor/bigcolor\_main.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://kimgeonung.github.io/assets/bigcolor/bigcolor_main.pdf "https://kimgeonung.github.io/assets/bigcolor/bigcolor_main.pdf")
*   Code:  [https://github.com/KIMGEONUNG/BigColor](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/KIMGEONUNG/BigColor "https://github.com/KIMGEONUNG/BigColor")

**Semantic-Sparse Colorization Network for Deep Exemplar-Based Colorization**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/820_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**ColorFormer: Image Colorization via Color Memory Assisted Hybrid-Attention Transformer**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3385_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**L-CoDer: Language-Based Colorization with Color-Object Decoupling Transformer**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2424_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Colorization for In Situ Marine Plankton Images**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6905_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

Image Matting – image matting
-----------------------------

**TransMatting: Enhancing Transparent Objects Matting with Transformers**

*   Paper:  [https://arxiv.org/abs/2208.03007](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.03007 "https://arxiv.org/abs/2208.03007")
*   Code:  [https://github.com/AceCHQ/TransMatting](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/AceCHQ/TransMatting "https://github.com/AceCHQ/TransMatting")

**One-Trimap Video Matting**

*   Paper:  [https://arxiv.org/abs/2207.13353](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.13353 "https://arxiv.org/abs/2207.13353")
*   Code:  [https://github.com/Hongje/OTVM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Hongje/OTVM "https://github.com/Hongje/OTVM")

Shadow Removal – shadow removal
-------------------------------

**Style-Guided Shadow Removal**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5580_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/jinwan1994/SG-ShadowNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jinwan1994/SG-ShadowNet "https://github.com/jinwan1994/SG-ShadowNet")

Image Compression – image compression
-------------------------------------

**Optimizing Image Compression via Joint Learning with Denoising**

*   Paper:  [https://arxiv.org/abs/2207.10869](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10869 "https://arxiv.org/abs/2207.10869")
*   Code:  [https://github.com/felixcheng97/DenoiseCompression](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/felixcheng97/DenoiseCompression "https://github.com/felixcheng97/DenoiseCompression")

**Implicit Neural Representations for Image Compression**

*   Paper:  [https://arxiv.org/abs/2112.04267](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.04267 "https://arxiv.org/abs/2112.04267")
*   Code: [https://github.com/YannickStruempler/inr\_based\_compression](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/YannickStruempler/inr_based_compression "https://github.com/YannickStruempler/inr_based_compression")

**Expanded Adaptive Scaling Normalization for End to End Image Compression**

*   Paper:  [https://arxiv.org/abs/2208.03049](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.03049 "https://arxiv.org/abs/2208.03049")

**Content-Oriented Learned Image Compression**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7542_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/lmijydyb/COLIC](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lmijydyb/COLIC "https://github.com/lmijydyb/COLIC")

**Contextformer: A Transformer with Spatio-Channel Attention for Context Modeling in Learned Image Compression**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6046_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Content Adaptive Latents and Decoder for Neural Image Compression**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4016_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

### Video Compression

**AlphaVC: High-Performance and Efficient Learned Video Compression**

*   Paper:  [https://arxiv.org/abs/2207.14678](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.14678 "https://arxiv.org/abs/2207.14678")

**CANF-VC: Conditional Augmented Normalizing Flows for Video Compression**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3904_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/NYCU-MAPL/CANF-VC](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/NYCU-MAPL/CANF-VC "https://github.com/NYCU-MAPL/CANF-VC")

**Neural Video Compression Using GANs for Detail Synthesis and Propagation**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4802_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

Image Quality Assessment – ​​image quality assessment
-----------------------------------------------------

**FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling**

*   Paper:  [https://arxiv.org/abs/2207.02595](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.02595 "https://arxiv.org/abs/2207.02595")
*   Code:  [https://github.com/TimothyHTimothy/FAST-VQA](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/TimothyHTimothy/FAST-VQA "https://github.com/TimothyHTimothy/FAST-VQA")

**Shift-tolerant Perceptual Similarity Metric**

*   Paper:  [https://arxiv.org/abs/2207.13686](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.13686 "https://arxiv.org/abs/2207.13686")
*   Code:  [GitHub – abhijay9/ShiftTolerant-LPIPS](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/abhijay9/ShiftTolerant-LPIPS/ "GitHub - abhijay9/ShiftTolerant-LPIPS")

**Telepresence Video Quality Assessment**

*   Paper:  [https://arxiv.org/abs/2207.09956](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.09956 "https://arxiv.org/abs/2207.09956")

**A Perceptual Quality Metric for Video Frame Interpolation**

*   Paper:  [https://arxiv.org/abs/2210.01879](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2210.01879 "https://arxiv.org/abs/2210.01879")
*   Code:  [https://github.com/hqqxyy/VFIPS](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/hqqxyy/VFIPS "https://github.com/hqqxyy/VFIPS")

Relighting/Delighting
---------------------

**Deep Portrait Delighting**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4581_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Geometry-Aware Single-Image Full-Body Human Relighting**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4385_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**NeRF for Outdoor Scene Relighting**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4998_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Physically-Based Editing of Indoor Scene Lighting from a Single Image**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1276_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

Style Transfer – style transfer
-------------------------------

**CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer**

*   Paper:  [https://arxiv.org/abs/2207.04808](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.04808 "https://arxiv.org/abs/2207.04808")
*   Code:  [https://github.com/JarrentWu1031/CCPL](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JarrentWu1031/CCPL "https://github.com/JarrentWu1031/CCPL")

**Image-Based CLIP-Guided Essence Transfer**

*   Paper:  [https://arxiv.org/abs/2110.12427](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2110.12427 "https://arxiv.org/abs/2110.12427")
*   Code:  [https://github.com/hila-chefer/TargetCLIP](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/hila-chefer/TargetCLIP "https://github.com/hila-chefer/TargetCLIP")

**Learning Graph Neural Networks for Image Style Transfer**

*   Paper:  [https://arxiv.org/abs/2207.11681](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.11681 "https://arxiv.org/abs/2207.11681")

**WISE: Whitebox Image Stylization by Example-based Learning**

*   Paper:  [https://arxiv.org/abs/2207.14606](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.14606 "https://arxiv.org/abs/2207.14606")
*   Code:  [https://github.com/winfried-loetzsch/wise](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/winfried-loetzsch/wise "https://github.com/winfried-loetzsch/wise")

**Language-Driven Artistic Style Transfer**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6627_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**MoDA: Map Style Transfer for Self-Supervised Domain Adaptation of Embodied Agents**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1762_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**JoJoGAN: One Shot Face Stylization**

*   Paper:  [https://arxiv.org/abs/2112.11641](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.11641 "https://arxiv.org/abs/2112.11641")
*   Code:  [https://github.com/mchong6/JoJoGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/mchong6/JoJoGAN "https://github.com/mchong6/JoJoGAN")

**EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer**

*   Paper:  [https://arxiv.org/abs/2207.09840](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.09840 "https://arxiv.org/abs/2207.09840")
*   Code:  [https://github.com/Chenyu-Yang-2000/EleGANt](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Chenyu-Yang-2000/EleGANt "https://github.com/Chenyu-Yang-2000/EleGANt")
*   Tags: Makeup Transfer

**RamGAN: Region Attentive Morphing GAN for Region-Level Makeup Transfer**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/803_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Makeup Transfer

Image Editing – image editing
-----------------------------

**Context-Consistent Semantic Image Editing with Style-Preserved Modulation**

*   Paper:  [https://arxiv.org/abs/2207.06252](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.06252 "https://arxiv.org/abs/2207.06252")
*   Code:  [https://github.com/WuyangLuo/SPMPGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/WuyangLuo/SPMPGAN "https://github.com/WuyangLuo/SPMPGAN")

**GAN with Multivariate Disentangling for Controllable Hair Editing**

*   Paper:  [https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/main/database/CtrlHair/CtrlHair.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/main/database/CtrlHair/CtrlHair.pdf "https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/main/database/CtrlHair/CtrlHair.pdf")
*   Code:  [https://github.com/XuyangGuo/CtrlHair](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/XuyangGuo/CtrlHair "https://github.com/XuyangGuo/CtrlHair")

**Paint2Pix: Interactive Painting based Progressive Image Synthesis and Editing**

*   Paper:  [https://arxiv.org/abs/2208.08092](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.08092 "https://arxiv.org/abs/2208.08092")
*   Code:  [https://github.com/1jsingh/paint2pix](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/1jsingh/paint2pix "https://github.com/1jsingh/paint2pix")

**High-fidelity GAN Inversion with Padding Space**

*   Paper:  [https://arxiv.org/abs/2203.11105](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.11105 "https://arxiv.org/abs/2203.11105")
*   Code:  [https://github.com/EzioBy/padinv](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/EzioBy/padinv "https://github.com/EzioBy/padinv")

**Text2LIVE: Text-Driven Layered Image and Video Editing**

*   Paper:  [https://arxiv.org/abs/2204.02491](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.02491 "https://arxiv.org/abs/2204.02491")
*   Code:  [https://github.com/omerbt/Text2LIVE](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/omerbt/Text2LIVE "https://github.com/omerbt/Text2LIVE")

**IntereStyle: Encoding an Interest Region for Robust StyleGAN Inversion**

*   Paper:  [https://arxiv.org/abs/2209.10811](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.10811 "https://arxiv.org/abs/2209.10811")

**Style Your Hair: Latent Optimization for Pose-Invariant Hairstyle Transfer via Local-Style-Aware Hair Alignment**

*   Paper:  [https://arxiv.org/abs/2208.07765](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.07765 "https://arxiv.org/abs/2208.07765")
*   Code:  [https://github.com/Taeu/Style-Your-Hair](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Taeu/Style-Your-Hair "https://github.com/Taeu/Style-Your-Hair")

**HairNet: Hairstyle Transfer with Pose Changes**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5227_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**End-to-End Visual Editing with a Generatively Pre-trained Artist**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/841_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**The Anatomy of Video Editing: A Dataset and Benchmark Suite for AI-Assisted Video Editing**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4736_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Scraping Textures from Natural Images for Synthesis and Editing**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2180_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language Guidance**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/8048_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Editing Out-of-Domain GAN Inversion via Differential Activations**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5504_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/HaoruiSong622/Editing-Out-of-Domain](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/HaoruiSong622/Editing-Out-of-Domain "https://github.com/HaoruiSong622/Editing-Out-of-Domain")

**ChunkyGAN: Real Image Inversion via Segments**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5092_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**FairStyle: Debiasing StyleGAN2 with Style Channel Manipulations**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7746_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/catlab-team/fairstyle](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/catlab-team/fairstyle "https://github.com/catlab-team/fairstyle")

**A Style-Based GAN Encoder for High Fidelity Reconstruction of Images and Videos**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2740_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/InterDigitalInc/FeatureStyleEncoder](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/InterDigitalInc/FeatureStyleEncoder "https://github.com/InterDigitalInc/FeatureStyleEncoder")

**Rayleigh EigenDirections (REDs): Nonlinear GAN latent space traversals for multidimensional features**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7277_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

Image Generation/Synthesis / Image-to-Image Translation – Image Generation/Synthesis/Translation
------------------------------------------------------------------------------------------------

### Text-to-Image / Text Guided / Multi-Modal

**TIPS: Text-Induced Pose Synthesis**

*   Paper:  [https://arxiv.org/abs/2207.11718](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.11718 "https://arxiv.org/abs/2207.11718")
*   Code:  [https://github.com/prasunroy/tips](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/prasunroy/tips "https://github.com/prasunroy/tips")

**TISE: A Toolbox for Text-to-Image Synthesis Evaluation**

*   Paper:  [https://arxiv.org/abs/2112.01398](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.01398 "https://arxiv.org/abs/2112.01398")
*   Code:  [https://github.com/VinAIResearch/tise-toolbox](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/VinAIResearch/tise-toolbox "https://github.com/VinAIResearch/tise-toolbox")

**Learning Visual Styles from Audio-Visual Associations**

*   Paper:  [https://arxiv.org/abs/2205.05072](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.05072 "https://arxiv.org/abs/2205.05072")
*   Code:  [https://github.com/Tinglok/avstyle](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Tinglok/avstyle "https://github.com/Tinglok/avstyle")

**Multimodal Conditional Image Synthesis with Product-of-Experts GANs**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3539_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Project:  [https://deepimagination.cc/PoE-GAN/](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://deepimagination.cc/PoE-GAN/ "https://deepimagination.cc/PoE-GAN/")

**NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5422_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Make-a-Scene: Scene-Based Text-to-Image Generation with Human Priors**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/993_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Trace Controlled Text to Image Generation**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1894_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Audio-Driven Stylized Gesture Generation with Flow-Based Model**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3948_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**No Token Left Behind: Explainability-Aided Image Classification and Generation**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2764_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

### Image-to-Image / Image Guided

**End-to-end Graph-constrained Vectorized Floorplan Generation with Panoptic Refinement**

*   Paper:  [https://arxiv.org/abs/2207.13268](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.13268 "https://arxiv.org/abs/2207.13268")

**ManiFest: Manifold Deformation for Few-shot Image Translation**

*   Paper:  [https://arxiv.org/abs/2111.13681](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2111.13681 "https://arxiv.org/abs/2111.13681")
*   Code:  [https://github.com/cv-rits/ManiFest](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/cv-rits/ManiFest "https://github.com/cv-rits/ManiFest")

**VecGAN: Image-to-Image Translation with Interpretable Latent Directions**

*   Paper:  [https://arxiv.org/abs/2207.03411](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.03411 "https://arxiv.org/abs/2207.03411")

**DynaST: Dynamic Sparse Transformer for Exemplar-Guided Image Generation**

*   Paper:  [https://arxiv.org/abs/2207.06124](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.06124 "https://arxiv.org/abs/2207.06124")
*   Code:  [https://github.com/Huage001/DynaST](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Huage001/DynaST "https://github.com/Huage001/DynaST")

**Cross Attention Based Style Distribution for Controllable Person Image Synthesis**

*   Paper:  [https://arxiv.org/abs/2208.00712](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.00712 "https://arxiv.org/abs/2208.00712")
*   Code:  [GitHub – xyzhouo/CASD](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/xyzhouo/CASD "GitHub - xyzhouo/CASD")

**Vector Quantized Image-to-Image Translation**

*   Paper:  [https://arxiv.org/abs/2207.13286](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.13286 "https://arxiv.org/abs/2207.13286")
*   Code:  [https://github.com/cyj407/VQ-I2I](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/cyj407/VQ-I2I "https://github.com/cyj407/VQ-I2I")

**URUST: Ultra-high-resolution unpaired stain transformation via Kernelized Instance Normalization**

*   Paper:  [https://arxiv.org/abs/2208.10730](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.10730 "https://arxiv.org/abs/2208.10730")
*   Code:  [https://github.com/Kaminyou/URUST](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Kaminyou/URUST "https://github.com/Kaminyou/URUST")

**General Object Pose Transformation Network from Unpaired Data**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5972_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code:  [https://github.com/suyukun666/UFO-PT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/suyukun666/UFO-PT "https://github.com/suyukun666/UFO-PT")

**Unpaired Image Translation via Vector Symbolic Architectures**

*   Paper:  [https://arxiv.org/abs/2209.02686](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.02686 "https://arxiv.org/abs/2209.02686")
*   Code:  [https://github.com/facebookresearch/vsait](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/facebookresearch/vsait "https://github.com/facebookresearch/vsait")

**Supervised Attribute Information Removal and Reconstruction for Image Manipulation**

*   Paper:  [https://arxiv.org/abs/2207.06555](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.06555 "https://arxiv.org/abs/2207.06555")
*   Code:  [https://github.com/NannanLi999/AIRR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/NannanLi999/AIRR "https://github.com/NannanLi999/AIRR")

**Bi-Level Feature Alignment for Versatile Image Translation and Manipulation**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3912_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Multi-Curve Translator for High-Resolution Photorealistic Image Translation**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1278_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**CoGS: Controllable Generation and Search from Sketch and Style**

*   Paper:  [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5160_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**AgeTransGAN for Facial Age Transformation with Rectified Performance Metrics**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1344_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/AvLab-CV/AgeTransGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/AvLab-CV/AgeTransGAN "https://github.com/AvLab-CV/AgeTransGAN")

### Others for image generation

**StyleLight: HDR Panorama Generation for Lighting Estimation and Editing**

*   Paper: [https://arxiv.org/abs/2207.14811](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.14811 "https://arxiv.org/abs/2207.14811")
*   Code: [https://github.com/Wanggcong/StyleLight](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Wanggcong/StyleLight "https://github.com/Wanggcong/StyleLight")

**Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling**

*   Paper: [https://arxiv.org/abs/2207.02196](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.02196 "https://arxiv.org/abs/2207.02196")
*   Code: [https://github.com/fudan-zvg/PDS](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fudan-zvg/PDS "https://github.com/fudan-zvg/PDS")

**GAN Cocktail: mixing GANs without dataset access**

*   Paper: [https://arxiv.org/abs/2106.03847](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2106.03847 "https://arxiv.org/abs/2106.03847")
*   Code: [https://github.com/omriav/GAN-cocktail](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/omriav/GAN-cocktail "https://github.com/omriav/GAN-cocktail")

**Compositional Visual Generation with Composable Diffusion Models**

*   Paper: [https://arxiv.org/abs/2206.01714](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.01714 "https://arxiv.org/abs/2206.01714")
*   Code: [https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch "https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch")

**Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation**

*   Paper: [https://arxiv.org/abs/2112.02450](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.02450 "https://arxiv.org/abs/2112.02450")
*   Code: [https://github.com/dzld00/Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/dzld00/Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation "https://github.com/dzld00/Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation")

**StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pretrained StyleGAN**

*   Paper: [https://arxiv.org/abs/2203.04036](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.04036 "https://arxiv.org/abs/2203.04036")
*   Code: [https://github.com/FeiiYin/StyleHEAT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/FeiiYin/StyleHEAT "https://github.com/FeiiYin/StyleHEAT")

**WaveGAN: An Frequency-aware GAN for High-Fidelity Few-shot Image Generation**

*   Paper: [https://arxiv.org/abs/2207.07288](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.07288 "https://arxiv.org/abs/2207.07288")
*   Code: [https://github.com/kobeshegu/ECCV2022\_WaveGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/kobeshegu/ECCV2022_WaveGAN "https://github.com/kobeshegu/ECCV2022_WaveGAN")

**FakeCLR: Exploring Contrastive Learning for Solving Latent Discontinuity in Data-Efficient GANs**

*   Paper: [https://arxiv.org/abs/2207.08630](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.08630 "https://arxiv.org/abs/2207.08630")
*   Code: [https://github.com/iceli1007/FakeCLR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/iceli1007/FakeCLR "https://github.com/iceli1007/FakeCLR")

**Auto-regressive Image Synthesis with Integrated Quantization**

*   Paper: [https://arxiv.org/abs/2207.10776](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10776 "https://arxiv.org/abs/2207.10776")
*   Code: [https://github.com/fnzhan/IQ-VAE](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fnzhan/IQ-VAE "https://github.com/fnzhan/IQ-VAE")

**PixelFolder: An Efficient Progressive Pixel Synthesis Network for Image Generation**

*   Paper: [https://arxiv.org/abs/2204.00833](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.00833 "https://arxiv.org/abs/2204.00833")
*   Code: [https://github.com/BlingHe/PixelFolder](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/BlingHe/PixelFolder "https://github.com/BlingHe/PixelFolder")

**DeltaGAN: Towards Diverse Few-shot Image Generation with Sample-Specific Delta**

*   Paper: [https://arxiv.org/abs/2207.10271](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10271 "https://arxiv.org/abs/2207.10271")
*   Code: [https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation "https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation")

**Generator Knows What Discriminator Should Learn in Unconditional GANs**

*   Paper: [https://arxiv.org/abs/2207.13320](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.13320 "https://arxiv.org/abs/2207.13320")
*   Code: [https://github.com/naver-ai/GGDR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/naver-ai/GGDR "https://github.com/naver-ai/GGDR")

**Hierarchical Semantic Regularization of Latent Spaces in StyleGANs**

*   Paper: [https://arxiv.org/abs/2208.03764](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.03764 "https://arxiv.org/abs/2208.03764")
*   Code: [https://drive.google.com/file/d/1gzHTYTgGBUlDWyN\_Z3ORofisQrHChg\_n/view](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://drive.google.com/file/d/1gzHTYTgGBUlDWyN_Z3ORofisQrHChg_n/view "https://drive.google.com/file/d/1gzHTYTgGBUlDWyN_Z3ORofisQrHChg_n/view")

**FurryGAN: High Quality Foreground-aware Image Synthesis**

*   Paper: [https://arxiv.org/abs/2208.10422](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.10422 "https://arxiv.org/abs/2208.10422")
*   Project: [FurryGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://jeongminb.github.io/FurryGAN/ "FurryGAN")

**Improving GANs for Long-Tailed Data through Group Spectral Regularization**

*   Paper: [https://arxiv.org/abs/2208.09932](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.09932 "https://arxiv.org/abs/2208.09932")
*   Code: [https://drive.google.com/file/d/1aG48i04Q8mOmD968PAgwEvPsw1zcS4Gk/view](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://drive.google.com/file/d/1aG48i04Q8mOmD968PAgwEvPsw1zcS4Gk/view "https://drive.google.com/file/d/1aG48i04Q8mOmD968PAgwEvPsw1zcS4Gk/view")

**Exploring Gradient-based Multi-directional Controls in GANs**

*   Paper: [https://arxiv.org/abs/2209.00698](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.00698 "https://arxiv.org/abs/2209.00698")
*   Code: [https://github.com/zikuncshelly/GradCtrl](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zikuncshelly/GradCtrl "https://github.com/zikuncshelly/GradCtrl")

**Improved Masked Image Generation with Token-Critic**

*   Paper: [https://arxiv.org/abs/2209.04439](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.04439 "https://arxiv.org/abs/2209.04439")

**Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation**

*   Paper: [https://arxiv.org/abs/2209.05968](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.05968 "https://arxiv.org/abs/2209.05968")
*   Project: [Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://eadcat.github.io/WSSN "Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation")

**Any-Resolution Training for High-Resolution Image Synthesis**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3693_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/chail/anyres-gan](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/chail/anyres-gan "https://github.com/chail/anyres-gan")

**BIPS: Bi-modal Indoor Panorama Synthesis via Residual Depth-Aided Adversarial Learning**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4327_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/chang9711/BIPS](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/chang9711/BIPS "https://github.com/chang9711/BIPS")

**Few-Shot Image Generation with Mixup-Based Distance Learning**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2709_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/reyllama/mixdl](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/reyllama/mixdl "https://github.com/reyllama/mixdl")

**StyleGAN-Human: A Data-Centric Odyssey of Human Generation**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3366_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/stylegan-human/StyleGAN-Human](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/stylegan-human/StyleGAN-Human "https://github.com/stylegan-human/StyleGAN-Human")

**StyleFace: Towards Identity-Disentangled Face Generation on Megapixels**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4255_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**Contrastive Learning for Diverse Disentangled Foreground Generation**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4323_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**BLT: Bidirectional Layout Transformer for Controllable Layout Generation**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7035_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/google-research/google-research/tree/master/layout-blt](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/google-research/google-research/tree/master/layout-blt "https://github.com/google-research/google-research/tree/master/layout-blt")

**Entropy-Driven Sampling and Training Scheme for Conditional Diffusion Generation**

*   Paper: [https://arxiv.org/abs/2206.11474](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.11474 "https://arxiv.org/abs/2206.11474")
*   Code: [https://github.com/ZGCTroy/ED-DPM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ZGCTroy/ED-DPM "https://github.com/ZGCTroy/ED-DPM")

**Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5081_ECCV_2022_paper.php "ECVA | European Computer Vision Association")

**DuelGAN: A Duel between Two Discriminators Stabilizes the GAN Training**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7143_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/UCSC-REAL/DuelGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/UCSC-REAL/DuelGAN "https://github.com/UCSC-REAL/DuelGAN")

### Video Generation

**Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer**

*   Paper: [https://arxiv.org/abs/2204.03638](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.03638 "https://arxiv.org/abs/2204.03638")
*   Code: [https://github.com/SongweiGe/TATS](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/SongweiGe/TATS "https://github.com/SongweiGe/TATS")

**Controllable Video Generation through Global and Local Motion Dynamics**

*   Paper: [https://arxiv.org/abs/2204.06558](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.06558 "https://arxiv.org/abs/2204.06558")
*   Code: [GitHub – Araachie/glass: Controllable Video Generation through Global and Local Motion Dynamics. In ECCV, 2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Araachie/glass "GitHub - Araachie/glass: Controllable Video Generation through Global and Local Motion Dynamics. In ECCV, 2022")

**Fast-Vid2Vid: Spatial-Temporal Compression for Video-to-Video Synthesis**

*   Paper: [https://arxiv.org/abs/2207.05049](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.05049 "https://arxiv.org/abs/2207.05049")
*   Code: [https://github.com/fast-vid2vid/fast-vid2vid](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fast-vid2vid/fast-vid2vid "https://github.com/fast-vid2vid/fast-vid2vid")

**Synthesizing Light Field Video from Monocular Video**

*   Paper: [https://arxiv.org/abs/2207.10357](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10357 "https://arxiv.org/abs/2207.10357")
*   Code: [https://github.com/ShrisudhanG/Synthesizing-Light-Field-Video-from-Monocular-Video](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ShrisudhanG/Synthesizing-Light-Field-Video-from-Monocular-Video "https://github.com/ShrisudhanG/Synthesizing-Light-Field-Video-from-Monocular-Video")

**StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation**

*   Paper: [https://arxiv.org/abs/2209.06192](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.06192 "https://arxiv.org/abs/2209.06192")
*   Code: [https://github.com/adymaharana/storydalle](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/adymaharana/storydalle "https://github.com/adymaharana/storydalle")

**Motion Transformer for Unsupervised Image Animation**

*   Paper:
*   Code: [https://github.com/JialeTao/MoTrans](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/JialeTao/MoTrans "https://github.com/JialeTao/MoTrans")

**Sound-Guided Semantic Video Generation**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5584_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/anonymous5584/sound-guided-semantic-video-generation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/anonymous5584/sound-guided-semantic-video-generation "https://github.com/anonymous5584/sound-guided-semantic-video-generation")

**Layered Controllable Video Generation**

*   Paper: [https://www.ecva.net/papers/eccv\_2022/papers\_ECCV/html/4847\_ECCV\_2022\_paper.php](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4847_ECCV_2022_paper.php "https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4847_ECCV_2022_paper.php")

**Diverse Generation from a Single Video Made Possible**

*   Paper: [https://arxiv.org/abs/2109.08591](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2109.08591 "https://arxiv.org/abs/2109.08591")
*   Code: [https://github.com/nivha/single\_video\_generation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/nivha/single_video_generation "https://github.com/nivha/single_video_generation")

**Semantic-Aware Implicit Neural Audio-Driven Video Portrait Generation**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/631_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/alvinliu0/SSP-NeRF](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/alvinliu0/SSP-NeRF "https://github.com/alvinliu0/SSP-NeRF")

**EAGAN: Efficient Two-Stage Evolutionary Architecture Search for GANs**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3419_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/marsggbo/EAGAN](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/marsggbo/EAGAN "https://github.com/marsggbo/EAGAN")

**BlobGAN: Spatially Disentangled Scene Representations**

*   Paper: [https://arxiv.org/abs/2205.02837](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2205.02837 "https://arxiv.org/abs/2205.02837")
*   Code: [https://github.com/dave-epstein/blobgan](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/dave-epstein/blobgan "https://github.com/dave-epstein/blobgan")

Others
------

**Learning Local Implicit Fourier Representation for Image Warping**

*   Paper: [https://ipl.dgist.ac.kr/LTEW.pdf](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://ipl.dgist.ac.kr/LTEW.pdf "https://ipl.dgist.ac.kr/LTEW.pdf")
*   Code: [https://github.com/jaewon-lee-b/ltew](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/jaewon-lee-b/ltew "https://github.com/jaewon-lee-b/ltew")
*   Tags: Image Warping

**Dress Code: High-Resolution Multi-Category Virtual Try-On**

*   Paper: [https://arxiv.org/abs/2204.08532](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2204.08532 "https://arxiv.org/abs/2204.08532")
*   Code: [https://github.com/aimagelab/dress-code](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/aimagelab/dress-code "https://github.com/aimagelab/dress-code")
*   Tags: Virtual Try-On

**High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions**

*   Paper: [https://arxiv.org/abs/2206.14180](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2206.14180 "https://arxiv.org/abs/2206.14180")
*   Code: [https://github.com/sangyun884/HR-VITON](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sangyun884/HR-VITON "https://github.com/sangyun884/HR-VITON")
*   Tags: Virtual Try-On

**Single Stage Virtual Try-on via Deformable Attention Flows**

*   Paper: [https://arxiv.org/abs/2207.09161](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.09161 "https://arxiv.org/abs/2207.09161")
*   Tags: Virtual Try-On

**Outpainting by Queries**

*   Paper: [https://arxiv.org/abs/2207.05312](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.05312 "https://arxiv.org/abs/2207.05312")
*   Code: [https://github.com/Kaiseem/QueryOTR](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Kaiseem/QueryOTR "https://github.com/Kaiseem/QueryOTR")
*   Tags: Outpainting

**Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal**

*   Paper: [https://arxiv.org/abs/2207.08178](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.08178 "https://arxiv.org/abs/2207.08178")
*   Code: [https://github.com/thinwayliu/Watermark-Vaccine](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/thinwayliu/Watermark-Vaccine "https://github.com/thinwayliu/Watermark-Vaccine")
*   Tags: Watermark Protection

**Efficient Meta-Tuning for Content-aware Neural Video Delivery**

*   Paper: [https://arxiv.org/abs/2207.09691](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.09691 "https://arxiv.org/abs/2207.09691")
*   Code: [https://github.com/Neural-video-delivery/EMT-Pytorch-ECCV2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Neural-video-delivery/EMT-Pytorch-ECCV2022 "https://github.com/Neural-video-delivery/EMT-Pytorch-ECCV2022")
*   Tags: Video Delivery

**Human-centric Image Cropping with Partition-aware and Content-preserving Features**

*   Paper: [https://arxiv.org/abs/2207.10269](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.10269 "https://arxiv.org/abs/2207.10269")
*   Code: [https://github.com/bcmi/Human-Centric-Image-Cropping](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/bcmi/Human-Centric-Image-Cropping "https://github.com/bcmi/Human-Centric-Image-Cropping")

**CelebV-HQ: A Large-Scale Video Facial Attributes Dataset**

*   Paper: [https://arxiv.org/abs/2207.12393](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.12393 "https://arxiv.org/abs/2207.12393")
*   Code: [https://github.com/CelebV-HQ/CelebV-HQ](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/CelebV-HQ/CelebV-HQ "https://github.com/CelebV-HQ/CelebV-HQ")
*   Tags: Dataset

**Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis**

*   Paper: [https://arxiv.org/abs/2207.11770](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.11770 "https://arxiv.org/abs/2207.11770")
*   Code: [https://github.com/sstzal/DFRF](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/sstzal/DFRF "https://github.com/sstzal/DFRF")
*   Tags: Talking Head Synthesis

**Responsive Listening Head Generation: A Benchmark Dataset and Baseline**

*   Paper: [https://arxiv.org/abs/2112.13548](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.13548 "https://arxiv.org/abs/2112.13548")
*   Code: [https://github.com/dc3ea9f/vico\_challenge\_baseline](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/dc3ea9f/vico_challenge_baseline "https://github.com/dc3ea9f/vico_challenge_baseline")

**Contrastive Monotonic Pixel-Level Modulation**

*   Paper: [https://arxiv.org/abs/2207.11517](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.11517 "https://arxiv.org/abs/2207.11517")
*   Code: [https://github.com/lukun199/MonoPix](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/lukun199/MonoPix "https://github.com/lukun199/MonoPix")

**AutoTransition: Learning to Recommend Video Transition Effects**

*   Paper: [https://arxiv.org/abs/2207.13479](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.13479 "https://arxiv.org/abs/2207.13479")
*   Code: [https://github.com/acherstyx/AutoTransition](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/acherstyx/AutoTransition "https://github.com/acherstyx/AutoTransition")

**Bringing Rolling Shutter Images Alive with Dual Reversed Distortion**

*   Paper: [https://arxiv.org/abs/2203.06451](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2203.06451 "https://arxiv.org/abs/2203.06451")
*   Code: [https://github.com/zzh-tech/Dual-Reversed-RS](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/zzh-tech/Dual-Reversed-RS "https://github.com/zzh-tech/Dual-Reversed-RS")

**Learning Object Placement via Dual-path Graph Completion**

*   Paper: [https://arxiv.org/abs/2207.11464](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2207.11464 "https://arxiv.org/abs/2207.11464")
*   Code: [https://github.com/bcmi/GracoNet-Object-Placement](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/bcmi/GracoNet-Object-Placement "https://github.com/bcmi/GracoNet-Object-Placement")

**DeepMCBM: A Deep Moving-camera Background Model**

*   Paper: [https://arxiv.org/abs/2209.07923](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.07923 "https://arxiv.org/abs/2209.07923")
*   Code: [https://github.com/BGU-CS-VIL/DeepMCBM](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/BGU-CS-VIL/DeepMCBM "https://github.com/BGU-CS-VIL/DeepMCBM")

**Mind the Gap in Distilling StyleGANs**

*   Paper: [https://arxiv.org/abs/2208.08840](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2208.08840 "https://arxiv.org/abs/2208.08840")
*   Code: [https://github.com/xuguodong03/StyleKD](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/xuguodong03/StyleKD "https://github.com/xuguodong03/StyleKD")

**StyleSwap: Style-Based Generator Empowers Robust Face Swapping**

*   Paper: [https://arxiv.org/abs/2209.13514](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2209.13514 "https://arxiv.org/abs/2209.13514")
*   Code: [https://github.com/Seanseattle/StyleSwap](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Seanseattle/StyleSwap "https://github.com/Seanseattle/StyleSwap")
*   Tags: Face Swapping

**Geometric Representation Learning for Document Image Rectification**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1698_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Code: [https://github.com/fh2019ustc/DocGeoNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/fh2019ustc/DocGeoNet "https://github.com/fh2019ustc/DocGeoNet")
*   Tags: Document Image Rectification

**Studying Bias in GANs through the Lens of Race**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2581_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: Racial Bias

**On the Robustness of Quality Measures for GANs**

*   Paper: [https://arxiv.org/abs/2201.13019](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2201.13019 "https://arxiv.org/abs/2201.13019")
*   Code: [https://github.com/MotasemAlfarra/R-FID-Robustness-of-Quality-Measures-for-GANs](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/MotasemAlfarra/R-FID-Robustness-of-Quality-Measures-for-GANs "https://github.com/MotasemAlfarra/R-FID-Robustness-of-Quality-Measures-for-GANs")

**TREND: Truncated Generalized Normal Density Estimation of Inception Embeddings for GAN Evaluation**

*   Paper: [ECVA | European Computer Vision Association](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3604_ECCV_2022_paper.php "ECVA | European Computer Vision Association")
*   Tags: GAN Evaluation

AAAI2022-Low-Level-Vision
=========================

Image Restoration 
------------------------

**Unsupervised Underwater Image Restoration: From a Homology Perspective**

*   Paper: [AAAI2022: Unsupervised Underwater Image Restoration: From a Homology Perspective](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai2078 "AAAI2022: Unsupervised Underwater Image Restoration: From a Homology Perspective")
*   Tags: Underwater Image Restoration

**Panini-Net: GAN Prior based Degradation-Aware Feature Interpolation for Face Restoration**

*   Paper: [AAAI2022: Panini-Net: GAN Prior Based Degradation-Aware Feature Interpolation for Face Restoration](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai4252 "AAAI2022: Panini-Net: GAN Prior Based Degradation-Aware Feature Interpolation for Face Restoration")
*   Code: [GitHub – wyhuai/Panini-Net: \[AAAI 2022\] Panini-Net: GAN Prior based Degradation-Aware Feature Interpolation for Face Restoration](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wyhuai/Panini-Net "GitHub - wyhuai/Panini-Net: [AAAI 2022] Panini-Net: GAN Prior based Degradation-Aware Feature Interpolation for Face Restoration")
*   Tags: Face Restoration

### Burst Restoration

**Zero-Shot Multi-Frame Image Restoration with Pre-Trained Siamese Transformers**

*   Paper: [AAAI2022: SiamTrans: Zero-Shot Multi-Frame Image Restoration with Pre-Trained Siamese Transformers](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai7488 "AAAI2022: SiamTrans: Zero-Shot Multi-Frame Image Restoration with Pre-Trained Siamese Transformers")
*   Code: [https://github.com/laulampaul/siamtrans](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/laulampaul/siamtrans "https://github.com/laulampaul/siamtrans")

### Video Restoration

**Transcoded Video Restoration by Temporal Spatial Auxiliary Network**

*   Paper: [AAAI2022: Transcoded Video Restoration by Temporal Spatial Auxiliary Network](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai12302 "AAAI2022: Transcoded Video Restoration by Temporal Spatial Auxiliary Network")
*   Tags: Transcoded Video Restoration

Super Resolution 
-----------------------

### Image Super Resolution

**SCSNet: An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-Resolution**

*   Paper: [AAAI2022: SCSNet: An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai528 "AAAI2022: SCSNet: An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-Resolution")

**Efficient Non-Local Contrastive Attention for Image Super-Resolution**

*   Paper: [https://arxiv.org/abs/2201.03794](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2201.03794 "https://arxiv.org/abs/2201.03794")
*   Code: [GitHub – Zj-BinXia/ENLCA: This project is official implementation of ‘Efficient Non-Local Contrastive Attention for Image Super-Resolution’, AAAI2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Zj-BinXia/ENLCA "GitHub - Zj-BinXia/ENLCA: This project is official implementation of 'Efficient Non-Local Contrastive Attention for Image Super-Resolution', AAAI2022")

**Best-Buddy GANs for Highly Detailed Image Super-Resolution**

*   Paper:  [AAAI2022: Best-Buddy GANs for Highly Detailed Image Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai137 "AAAI2022: Best-Buddy GANs for Highly Detailed Image Super-Resolution")
*   Tags: GAN

**Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution**

*   Paper:  [AAAI2022: Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai1194 "AAAI2022: Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution")
*   Tags: Text SR

**Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-Based Super-Resolution**

*   Paper:  [AAAI2022: Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-Based Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai390 "AAAI2022: Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-Based Super-Resolution")
*   Code:  [GitHub – Zj-BinXia/AMSA: This project is the official implementation of 'Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-based Super-Resolution', AAAI2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Zj-BinXia/AMSA "GitHub - Zj-BinXia/AMSA: This project is the official implementation of 'Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-based Super-Resolution', AAAI2022")
*   Tags: Reference-Based SR

**Detail-Preserving Transformer for Light Field Image Super-Resolution**

*   Paper:  [AAAI2022: Detail-Preserving Transformer for Light Field Image Super-Resolution](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai3550 "AAAI2022: Detail-Preserving Transformer for Light Field Image Super-Resolution")
*   Tags: Light Field

Denoising – denoising
---------------------

### Image Denoising

**Generative Adaptive Convolutions for Real-World Noisy Image Denoising**

*   Paper:  [AAAI2022: Generative Adaptive Convolutions for Real-World Noisy Image Denoising](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai4230 "AAAI2022: Generative Adaptive Convolutions for Real-World Noisy Image Denoising")

### Video Denoising

**ReMoNet: Recurrent Multi-Output Network for Efficient Video Denoising**

*   Paper:  [AAAI2022: ReMoNet: Recurrent Multi-Output Network for Efficient Video Denoising](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai6295 "AAAI2022: ReMoNet: Recurrent Multi-Output Network for Efficient Video Denoising")

Deblurring – Deblurring
-----------------------

### Video Deblurring

**Deep Recurrent Neural Network with Multi-Scale Bi-Directional Propagation for Video Deblurring**

*   Paper:  [AAAI2022: Deep Recurrent Neural Network with Multi-Scale Bi-Directional Propagation for Video Deblurring](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai3124 "AAAI2022: Deep Recurrent Neural Network with Multi-Scale Bi-Directional Propagation for Video Deblurring")

Deraining – deraining
---------------------

**Online-Updated High-Order Collaborative Networks for Single Image Deraining**

*   Paper:  [AAAI2022: ReMoNet: Recurrent Multi-Output Network for Efficient Video Denoising](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai6295 "AAAI2022: ReMoNet: Recurrent Multi-Output Network for Efficient Video Denoising")

**Close the Loop: A Unified Bottom-up and Top-down Paradigm for Joint Image Deraining and Segmentation**

*   Paper:  [AAAI2022: Close the Loop: A Unified Bottom-up and Top-down Paradigm for Joint Image Deraining and Segmentation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai678 "AAAI2022: Close the Loop: A Unified Bottom-up and Top-down Paradigm for Joint Image Deraining and Segmentation")
*   Tags: Joint Image Deraining and Segmentation

Dehazing – to fog
-----------------

**Uncertainty-Driven Dehazing Network**

*   Paper:  [AAAI2022: Uncertainty-Driven Dehazing Network](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai2838 "AAAI2022: Uncertainty-Driven Dehazing Network")

Demosaicing – Demosaicing
-------------------------

**Deep Spatial Adaptive Network for Real Image Demosaicing**

*   Paper:  [AAAI2022: Deep Spatial Adaptive Network for Real Image Demosaicing](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai2170 "AAAI2022: Deep Spatial Adaptive Network for Real Image Demosaicing")

HDR Imaging / Multi-Exposure Image Fusion – HDR image generation / multi-exposure image fusion
----------------------------------------------------------------------------------------------

**TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework Using Self-Supervised Multi-Task Learning**

*   Paper:  [https://arxiv.org/abs/2112.01030](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.01030 "https://arxiv.org/abs/2112.01030")
*   Code:  [https://github.com/miccaiif/TransMEF](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/miccaiif/TransMEF "https://github.com/miccaiif/TransMEF")

Image Enhancement – ​​image enhancement
---------------------------------------

### Low-Light Image Enhancement

**Low-Light Image Enhancement with Normalizing Flow**

*   Paper:  [https://arxiv.org/abs/2109.05923](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2109.05923 "https://arxiv.org/abs/2109.05923")
*   Code:  [https://github.com/wyf0912/LLFlow](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wyf0912/LLFlow "https://github.com/wyf0912/LLFlow")

**Degrade is Upgrade: Learning Degradation for Low-light Image Enhancement**

*   Paper:  [AAAI2022: Degrade is Upgrade: Learning Degradation for Low-light Image Enhancement](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai841 "AAAI2022: Degrade is Upgrade: Learning Degradation for Low-light Image Enhancement")

**Semantically Contrastive Learning for Low-Light Image Enhancement**

*   Paper:  [AAAI2022: Semantically Contrastive Learning for Low-Light Image Enhancement](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai4218 "AAAI2022: Semantically Contrastive Learning for Low-Light Image Enhancement")
*   Tags: contrastive learning

Image Matting – image matting
-----------------------------

**MODNet: Trimap-Free Portrait Matting in Real Time**

*   Paper:  [https://arxiv.org/abs/2011.11961](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2011.11961 "https://arxiv.org/abs/2011.11961")
*   Code:  [https://github.com/ZHKKKe/MODNet](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/ZHKKKe/MODNet "https://github.com/ZHKKKe/MODNet")

Shadow Removal – shadow removal
-------------------------------

**Efficient Model-Driven Network for Shadow Removal**

*   Paper:  [AAAI2022: Efficient Model-Driven Network for Shadow Removal](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai196 "AAAI2022: Efficient Model-Driven Network for Shadow Removal")

Image Compression – image compression
-------------------------------------

**Towards End-to-End Image Compression and Analysis with Transformers**

*   Paper:  [https://arxiv.org/abs/2112.09300](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2112.09300 "https://arxiv.org/abs/2112.09300")
*   Code:  [https://github.com/BYchao100/Towards-Image-Compression-and-Analysis-with-Transformers](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/BYchao100/Towards-Image-Compression-and-Analysis-with-Transformers "https://github.com/BYchao100/Towards-Image-Compression-and-Analysis-with-Transformers")
*   Tags: Transformer

**OoDHDR-Codec: Out-of-Distribution Generalization for HDR Image Compression**

*   Paper:  [AAAI2022: OoDHDR-Codec: Out-of-Distribution Generalization for HDR Image Compression](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai8610 "AAAI2022: OoDHDR-Codec: Out-of-Distribution Generalization for HDR Image Compression")

**Two-Stage Octave Residual Network for End-to-End Image Compression**

*   Paper:  [AAAI2022: Two-Stage Octave Residual Network for End-to-End Image Compression](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai4043 "AAAI2022: Two-Stage Octave Residual Network for End-to-End Image Compression")

Image Quality Assessment – ​​image quality assessment
-----------------------------------------------------

**Content-Variant Reference Image Quality Assessment via Knowledge Distillation**

*   Paper:  [AAAI2022: Content-Variant Reference Image Quality Assessment via Knowledge Distillation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai1344 "AAAI2022: Content-Variant Reference Image Quality Assessment via Knowledge Distillation")

**Perceptual Quality Assessment of Omnidirectional Images**

*   Paper:  [AAAI2022: Perceptual Quality Assessment of Omnidirectional Images](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai4008 "AAAI2022: Perceptual Quality Assessment of Omnidirectional Images")
*   Tags: Omnidirectional Images

Style Transfer – style transfer
-------------------------------

**Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization**

*   Paper:  [https://arxiv.org/abs/2103.11784](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://arxiv.org/abs/2103.11784 "https://arxiv.org/abs/2103.11784")
*   Code:  [GitHub – czczup/URST: \[AAAI 2022\] Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/czczup/URST "GitHub - czczup/URST: [AAAI 2022] Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization")

**Deep Translation Prior: Test-Time Training for Photorealistic Style Transfer**

*   Paper:  [AAAI2022: Deep Translation Prior: Test-Time Training for Photorealistic Style Transfer](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai1958 "AAAI2022: Deep Translation Prior: Test-Time Training for Photorealistic Style Transfer")

Image Editing – image editing
-----------------------------

Image Generation/Synthesis / Image-to-Image Translation – Image Generation/Synthesis/Translation
------------------------------------------------------------------------------------------------

**SSAT: A Symmetric Semantic-Aware Transformer Network for Makeup Transfer and Removal**

*   Paper:  [AAAI2022: SSAT: A Symmetric Semantic-Aware Transformer Network for Makeup Transfer and Removal](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai934 "AAAI2022: SSAT: A Symmetric Semantic-Aware Transformer Network for Makeup Transfer and Removal")
*   Code:  [https://github.com/Snowfallingplum/SSAT](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Snowfallingplum/SSAT "https://github.com/Snowfallingplum/SSAT")
*   Tags: Makeup Transfer and Removal

**Assessing a Single Image in Reference-Guided Image Synthesis**

*   Paper:  [AAAI2022: Assessing a Single Image in Reference-Guided Image Synthesis](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai1241 "AAAI2022: Assessing a Single Image in Reference-Guided Image Synthesis")

**Interactive Image Generation with Natural-Language Feedback**

*   Paper:  [AAAI2022: Interactive Image Generation with Natural-Language Feedback](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai7081 "AAAI2022: Interactive Image Generation with Natural-Language Feedback")

**PetsGAN: Rethinking Priors for Single Image Generation**

*   Paper:  [AAAI2022: PetsGAN: Rethinking Priors for Single Image Generation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai2865 "AAAI2022: PetsGAN: Rethinking Priors for Single Image Generation")

**Pose Guided Image Generation from Misaligned Sources via Residual Flow Based Correction**

*   Paper:  [AAAI2022: Pose Guided Image Generation from Misaligned Sources via Residual Flow Based Correction](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai4099 "AAAI2022: Pose Guided Image Generation from Misaligned Sources via Residual Flow Based Correction")

**Hierarchical Image Generation via Transformer-Based Sequential Patch Selection**

*   Paper:  [AAAI2022: Hierarchical Image Generation via Transformer-Based Sequential Patch Selection](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai557 "AAAI2022: Hierarchical Image Generation via Transformer-Based Sequential Patch Selection")

**Style-Guided and Disentangled Representation for Robust Image-to-Image Translation**

*   Paper:  [AAAI2022: Style-Guided and Disentangled Representation for Robust Image-to-Image Translation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai3727 "AAAI2022: Style-Guided and Disentangled Representation for Robust Image-to-Image Translation")

**OA-FSUI2IT: A Novel Few-Shot Cross Domain Object Detection Framework with Object-Aware Few-shot Unsupervised Image-to-Image Translation**

*   Paper:  [AAAI2022: OA-FSUI2IT: A Novel Few-Shot Cross Domain Object Detection Framework with Object-Aware Few-shot Unsupervised Image-to-Image Translation](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai2213 "AAAI2022: OA-FSUI2IT: A Novel Few-Shot Cross Domain Object Detection Framework with Object-Aware Few-shot Unsupervised Image-to-Image Translation")
*   Code:  [https://github.com/emdata-ailab/FSCD-Det](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/emdata-ailab/FSCD-Det "https://github.com/emdata-ailab/FSCD-Det")
*   Tags: Image-to-Image Translation used for Object Detection

### Video Generation

**Learning Temporally and Semantically Consistent Unpaired Video-to-Video Translation through Pseudo-Supervision from Synthetic Optical Flow**

*   Paper:  [AAAI2022: Learning Temporally and Semantically Consistent Unpaired Video-to-Video Translation through Pseudo-Supervision from Synthetic Optical Flow](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://aaai-2022.virtualchair.net/poster_aaai4610 "AAAI2022: Learning Temporally and Semantically Consistent Unpaired Video-to-Video Translation through Pseudo-Supervision from Synthetic Optical Flow")
*   Code:  [GitHub – wangkaihong/Unsup\_Recycle\_GAN: Code for “Learning Temporally and Semantically Consistent Unpaired Video-to-video Translation Through Pseudo-Supervision From Synthetic Optical Flow”, AAAI 2022](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/wangkaihong/Unsup_Recycle_GAN "GitHub - wangkaihong/Unsup_Recycle_GAN: Code for")

Refer to
========

[What are low-level and high-level tasks\_low-level tasks\_WTHunt's Blog-CSDN Blog](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://blog.csdn.net/qq_20880415/article/details/117225213 "What are low-level and high-level tasks_low-level tasks_WTHunt's Blog-CSDN Blog")

[What is the prospect of low-level vision in the CV field? – Zhihu (zhihu.com)](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://www.zhihu.com/question/467432767 "What is the prospect of low-level vision in the CV field?  - Zhihu (zhihu.com)")

[GitHub – DarrenPan/Awesome-CVPR2023-Low-Level-Vision: A Collection of Papers and Codes in CVPR2023/2022 about low level vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-CVPR2023-Low-Level-Vision "GitHub - DarrenPan/Awesome-CVPR2023-Low-Level-Vision: A Collection of Papers and Codes in CVPR2023/2022 about low level vision")

*   [Awesome-CVPR2022-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-CVPR2023-Low-Level-Vision/blob/main/CVPR2022-Low-Level-Vision.md "Awesome-CVPR2022-Low-Level-Vision")
*   [Awesome-ECCV2022-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-ECCV2022-Low-Level-Vision "Awesome-ECCV2022-Low-Level-Vision")
*   [Awesome-AAAI2022-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-AAAI2022-Low-Level-Vision "Awesome-AAAI2022-Low-Level-Vision")
*   [Awesome-NeurIPS2021-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/DarrenPan/Awesome-NeurIPS2021-Low-Level-Vision "Awesome-NeurIPS2021-Low-Level-Vision")
*   [Awesome-ICCV2021-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Kobaayyy/Awesome-ICCV2021-Low-Level-Vision "Awesome-ICCV2021-Low-Level-Vision")
*   [Awesome-CVPR2021/CVPR2020-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Kobaayyy/Awesome-CVPR2021-CVPR2020-Low-Level-Vision "Awesome-CVPR2021/CVPR2020-Low-Level-Vision")
*   [Awesome-ECCV2020-Low-Level-Vision](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/Kobaayyy/Awesome-ECCV2020-Low-Level-Vision "Awesome-ECCV2020-Low-Level-Vision")







