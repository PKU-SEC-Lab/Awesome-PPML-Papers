# Privacy-Preserving Machine Learning (PPML) Survey

This repo contains the papers mentioned in paper XXX.

## Protocol-Level Optimization


## Model-Level Optimization

### Linear Layer Optimization

* [CCS 2021] COINN: Crypto/ML Codesign for Oblivious Inference via Neural Networks [[paper](https://dl.acm.org/doi/pdf/10.1145/3460120.3484797)] [[code](https://github.com/ACESLabUCSD/COINN)]
* [ASIA CCS 2022] Hunter: HE-Friendly Structured Pruning for Efficient Privacy-Preserving Deep Learning [[paper](https://dl.acm.org/doi/pdf/10.1145/3488932.3517401)]
* [arXiv 2022] Efficient ML Models for Practical Secure Inference [[paper](https://arxiv.org/pdf/2209.00411)]
* [NeurIPS 2023] CoPriv: Network/Protocol Co-Optimization for Communication-Efficient Private Inference [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f96839fc751b67492e17e70f5c9730e4-Paper-Conference.pdf)]
* [ICCV 2023] MPCViT: Searching for Accurate and Efficient MPC-Friendly Vision Transformer with Heterogeneous Attention [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_MPCViT_Searching_for_Accurate_and_Efficient_MPC-Friendly_Vision_Transformer_with_ICCV_2023_paper.pdf)] [[code](https://github.com/PKU-SEC-Lab/mpcvit)]
* [NeurIPS 2024] PrivCirNet: Efficient Private Inference via Block Circulant Transformation [[paper](https://eprint.iacr.org/2024/2008.pdf)] [[code](https://github.com/Tianshi-Xu/PrivCirNet)]
* [arXiv 2024] AERO: Softmax-Only LLMs for Efficient Private Inference [[paper](https://arxiv.org/pdf/2410.13060)]
* [arXiv 2024] EQO: Exploring Ultra-Efficient Private Inference with Winograd-Based Protocol and Quantization Co-Optimization [[paper](https://arxiv.org/pdf/2404.09404v1)]

### Non-Linear ReLU/GeLU Optimization

* [Security 2020] DELPHI: A Cryptographic Inference Service for Neural Networks [[paper](https://www.usenix.org/system/files/sec20spring_mishra_prepub.pdf)]
* [NeurIPS 2020] CryptoNAS: Private Inference on a ReLU Budget [[paper](https://papers.neurips.cc/paper_files/paper/2020/file/c519d47c329c79537fbb2b6f1c551ff0-Paper.pdf)]
* [ICML 2021] DeepReDuce: ReLU Reduction for Fast Private Inference [[paper](https://proceedings.mlr.press/v139/jha21a/jha21a.pdf)]
* [ICLR 2020] Safenet: A secure, accurate and fast neural network inference [[paper](https://openreview.net/pdf?id=Cz3dbFm5u-)]
* [IEEE Security & Privacy] Sphynx: A Deep Neural Network Design for Private Inference [[paper](https://arxiv.org/pdf/2106.11755)]
* [NeurIPS 2021] Circa: Stochastic ReLUs for Private Deep Learning [[paper](https://proceedings.neurips.cc/paper/2021/file/11eba2991cc62daa4a85be5c0cfdae97-Paper.pdf)]
* [ICML 2022] Selective Network Linearization for Efficient Private Inference [[paper](https://proceedings.mlr.press/v162/cho22a/cho22a.pdf)] [[code](https://github.com/NYU-DICE-Lab/selective_network_linearization)]
* [arXiv 2022] AESPA: Accuracy Preserving Low-degree Polynomial Activation for Fast Private Inference [[paper](https://arxiv.org/pdf/2201.06699)]
* [arXiv 2023] RRNet: Towards ReLU-Reduced Neural Network for Two-party Computation Based Private Inference [[paper](https://arxiv.org/pdf/2302.02292)]
* [ACL 2022] THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption [[paper](https://aclanthology.org/2022.findings-acl.277.pdf)]
* [ICLR 2023] MPCFormer: fast, performant and private Transformer inference with MPC [[paper](https://arxiv.org/pdf/2211.01452)] [[code](https://github.com/DachengLi1/MPCFormer)]
* [ICCV 2023] MPCViT: Searching for Accurate and Efficient MPC-Friendly Vision Transformer with Heterogeneous Attention [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_MPCViT_Searching_for_Accurate_and_Efficient_MPC-Friendly_Vision_Transformer_with_ICCV_2023_paper.pdf)] [[code](https://github.com/PKU-SEC-Lab/mpcvit)]
* [arXiv 2023] PRIVIT: VISION TRANSFORMERS FOR FAST PRIVATE INFERENCE [[paper](https://arxiv.org/pdf/2310.04604)] [[code](https://github.com/NYU-DICE-Lab/privit)]
* [NeurIPS 2023] CoPriv: Network/Protocol Co-Optimization for Communication-Efficient Private Inference [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f96839fc751b67492e17e70f5c9730e4-Paper-Conference.pdf)]
* [ICCV 2023] AutoReP: Automatic ReLU Replacement for Fast Private Network Inference [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Peng_AutoReP_Automatic_ReLU_Replacement_for_Fast_Private_Network_Inference_ICCV_2023_paper.pdf)] [[code](https://github.com/HarveyP123/AutoReP)]
* [ICLR 2023] Learning to Linearize Deep Neural Networks for Secure and Efficient Private Inference [[paper](https://arxiv.org/pdf/2301.09254)]
* [DAC 2023] PASNet: Polynomial Architecture Search Framework for Two-party Computation-based Secure Neural Network Deployment [[paper](https://arxiv.org/pdf/2306.15513)] [[code](https://github.com/harveyp123/PASNet-DAC2023)]
* [arXiv 2023] Securing Neural Networks with Knapsack Optimization [[paper](https://arxiv.org/pdf/2304.10442v2)] [[code](https://github.com/yg320/secure_inference)]
* [arXiv 2023] LLMs Can Understand Encrypted Prompt: Towards Privacy-Computing Friendly Transformers [[paper](https://arxiv.org/pdf/2305.18396)]
* [arXiv 2023] Optimized Layerwise Approximation for Efficient Private Inference on Fully Homomorphic Encryption [[paper](https://arxiv.org/pdf/2310.10349)]
* [Security 2024] Fast and Private Inference of Deep Neural Networks by Co-designing Activation Functions [[paper](https://www.usenix.org/system/files/sec24summer-prepub-373-diaa.pdf)] [[code](https://github.com/LucasFenaux/PILLAR-ESPN)]
* [ICCAD 2023] RNA-ViT: Reduced-Dimension Approximate Normalized Attention Vision Transformers for Latency Efficient Private Inference [[paper](https://ieeexplore.ieee.org/abstract/document/10323702)]
* [arXiv 2024] AERO: Softmax-Only LLMs for Efficient Private Inference [[paper](https://arxiv.org/pdf/2410.13060)]
* [ICML 2024] Seesaw: Compensating for Nonlinear Reduction with Linear Computations for Private Inference [[paper](https://proceedings.mlr.press/v235/li24cj.html)]
* [TMLR 2024] DeepReShape: Redesigning Neural Networks for Efficient Private Inference [[paper](https://arxiv.org/pdf/2304.10593)]
* [ICML 2024] Ditto: Quantization-aware Secure Inference of Transformers upon MPC [[paper](https://arxiv.org/pdf/2405.05525)]

### Non-Linear Softmax Optimization

* [ACL 2022] THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption [[paper](https://aclanthology.org/2022.findings-acl.277.pdf)]
* [ICLR 2023] MPCFormer: fast, performant and private Transformer inference with MPC [[paper](https://arxiv.org/pdf/2211.01452)] [[code](https://github.com/DachengLi1/MPCFormer)]
* [ICCV 2023] MPCViT: Searching for Accurate and Efficient MPC-Friendly Vision Transformer with Heterogeneous Attention [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_MPCViT_Searching_for_Accurate_and_Efficient_MPC-Friendly_Vision_Transformer_with_ICCV_2023_paper.pdf)] [[code](https://github.com/PKU-SEC-Lab/mpcvit)]
* [arXiv 2023] PRIVIT: VISION TRANSFORMERS FOR FAST PRIVATE INFERENCE [[paper](https://arxiv.org/pdf/2310.04604)] [[code](https://github.com/NYU-DICE-Lab/privit)]
* [ICCV 2023] SAL-ViT: Towards Latency Efficient Private Inference on ViT using Selective Attention Search with a Learnable Softmax Approximation [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_SAL-ViT_Towards_Latency_Efficient_Private_Inference_on_ViT_using_Selective_ICCV_2023_paper.pdf)]
* [ICCAD 2023] RNA-ViT: Reduced-Dimension Approximate Normalized Attention Vision Transformers for Latency Efficient Private Inference [[paper](https://ieeexplore.ieee.org/abstract/document/10323702)]
* [Journal of Cryptographic Engineering 2025] MLFormer: a high performance MPC linear inference framework for transformers [[paper](https://link.springer.com/article/10.1007/s13389-024-00365-1)]
* [arXiv 2024] MPC-Minimized Secure LLM Inference [[paper](https://arxiv.org/pdf/2408.03561)]
* [arXiv 2024] Power-Softmax: Towards Secure LLM Inference over Encrypted Data [[paper](https://arxiv.org/pdf/2410.09457)]
* [ICML 2024] Converting Transformers to Polynomial Form for Secure Inference Over Homomorphic Encryption [[paper](https://arxiv.org/pdf/2311.08610)]
* [ACL 2024] SecFormer: Fast and Accurate Privacy-Preserving Inference for Transformer Models via SMPC [[paper](https://aclanthology.org/2024.findings-acl.790.pdf)] [[code](https://github.com/jinglong696/SecFormer)]
* [arXiv 2025] MPCache: MPC-Friendly KV Cache Eviction for Efficient Private Large Language Model Inference [[paper](https://arxiv.org/pdf/2501.06807)]
* [ICLR 2025] CipherPrune: Efficient and Scalable Private Transformer Inference [[paper](https://arxiv.org/pdf/2502.16782)]

### PPML-Friendly Quantization Optimization

* [CCS 2021] COINN: Crypto/ML Codesign for Oblivious Inference via Neural Networks [[paper](https://dl.acm.org/doi/pdf/10.1145/3460120.3484797)] [[code](https://github.com/ACESLabUCSD/COINN)]
* [Security 2019] XONN: XNOR-based Oblivious Deep Neural Network Inference [[paper](https://eprint.iacr.org/2019/171.pdf)]
* [ICCAD 2024] PrivQuant: Communication-Efficient Private Inference with Quantized Network/Protocol Co-Optimization [[paper](https://eprint.iacr.org/2024/2021.pdf)]
* [arXiv 2024] EQO: Exploring Ultra-Efficient Private Inference with Winograd-Based Protocol and Quantization Co-Optimization [[paper](https://arxiv.org/pdf/2404.09404v1)]
* [ICML 2024] Ditto: Quantization-aware Secure Inference of Transformers upon MPC [[paper](https://arxiv.org/pdf/2405.05525)]

## System-Level Optimization

### Compiler

* [CF 2019] nGraph-HE: A Graph Compiler for Deep Learning on Homomorphically Encrypted Data [[paper](https://dl.acm.org/doi/10.1145/3310273.3323047)] [[code](https://github.com/intel/he-transformer)]
* [WAHC 2019] nGraph-HE2: A High-Throughput Framework for Neural Network Inference on Encrypted Data [[paper](https://dl.acm.org/doi/10.1145/3338469.3358944)] [[code](https://github.com/intel/he-transformer)]
* [PLDI 2019] CHET: An Optimizing Compiler for Fully Homomorphic Neural Network Inferencing [[paper](https://dl.acm.org/doi/10.1145/3314221.3314628)]
* [PLDI 2020] Optimizing homomorphic evaluation circuits by program synthesis and term rewriting [[paper](https://dl.acm.org/doi/10.1145/3385412.3385996)] [[code](https://github.com/dklee0501/Lobster)]
* [PLDI 2020] EVA: An Encrypted Vector Arithmetic Language and Compiler for Efficient Homomorphic Computation [[paper](https://dl.acm.org/doi/abs/10.1145/3385412.3386023)] [[code](https://github.com/microsoft/EVA)]
* [WAHC 2020] Concete: Concrete Operates on Ciphertexts Rapidly by Extending TFHE [[paper](https://inria.hal.science/hal-03926650)] [[code](https://github.com/zama-ai/concrete)]
* [ArXiv 2021] A General Purpose Transpiler for Fully Homomorphic Encryption [[paper](https://eprint.iacr.org/2021/811)] [[code](https://github.com/google/fully-homomorphic-encryption/tree/main/transpiler)]
* [PLDI 2021] Porcupine: A Synthesizing Compiler for Vectorized Homomorphic Encryption [[paper](https://dl.acm.org/doi/10.1145/3453483.3454050)]
* [CGO 2022] HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption Compiler [[paper](https://dl.acm.org/doi/10.1109/CGO53902.2022.9741265)] [[code](https://github.com/corelab-src/elasm)]
* [PoPETs 2023] HElayers: A Tile Tensors Framework for Large Neural Networks on Encrypted Data [[paper](https://doi.org/10.56553/popets-2023-0020)] [[code](https://github.com/IBM/helayers)]
* [ASPLOS 2023] Coyote: A Compiler for Vectorizing Encrypted Arithmetic Circuits [[paper](https://dl.acm.org/doi/10.1145/3582016.3582057)] [[code](https://github.com/raghav198/coyote)]
* [Security 2023] ELASM: Error-Latency-Aware Scale Management for Fully Homomorphic Encryption [[paper](https://dl.acm.org/doi/10.5555/3620237.3620500)]
* [Security 2023] HECO: Fully Homomorphic Encryption Compiler [[paper](https://dl.acm.org/doi/10.5555/3620237.3620501)] [[code](https://github.com/MarbleHE/HECO)]
* [PLDI 2024] A Tensor Compiler with Automatic Data Packing for Simple and Efficient Fully Homomorphic Encryption [[paper](https://dl.acm.org/doi/10.1145/3656382)] [[code](https://github.com/fhelipe-compiler/fhelipe)]
* [Security 2024] DaCapo: Automatic Bootstrapping Management for Efficient Fully Homomorphic Encryption [[paper](https://dl.acm.org/doi/10.5555/3698900.3699291)] [[code](https://github.com/corelab-src/dacapo)]
* [CGO 2025] ANT-ACE: An FHE Compiler Framework for Automating Neural Network Inference [[paper](https://dl.acm.org/doi/10.1145/3696443.3708924)] [[code](https://github.com/ant-research/ace-compiler)]
* [ASPLOS 2025] HALO: Loop-aware Bootstrapping Management for Fully Homomorphic Encryption [[paper](https://dl.acm.org/doi/10.1145/3669940.3707275)]
* [ASPLOS 2025] ReSBM: Region-based Scale and Minimal-Level Bootstrapping Management for FHE via Min-Cut [[paper](https://dl.acm.org/doi/10.1145/3669940.3707276)]
* [ASPLOS 2025] Orion: A Fully Homomorphic Encryption Framework for Deep Learning [[paper](https://dl.acm.org/doi/10.1145/3676641.3716008)] [[code](https://github.com/baahl-nyu/orion)]


### GPU Optimization

* [HPCA 2025] WarpDrive: GPU-Based Fully Homomorphic Encryption Acceleration Leveraging Tensor and CUDA Cores [[paper](https://ieeexplore.ieee.org/document/10946827)]
* [HPCA 2025] Anaheim: Architecture and Algorithms for Processing Fully Homomorphic Encryption in Memory [[paper](https://ieeexplore.ieee.org/document/10946801)]
* [TCHES 2025] VeloFHE: GPU Acceleration for FHEW and TFHE Bootstrapping [[paper](https://tches.iacr.org/index.php/TCHES/article/view/12211)]
* [TCHES 2025] GPU Acceleration for FHEW/TFHE Bootstrapping [[paper](https://tches.iacr.org/index.php/TCHES/article/view/11931)]
* [PoPETS 2025] Hardware-Accelerated Encrypted Execution of General-Purpose Applications [[paper](https://petsymposium.org/popets/2025/popets-2025-0039.php)] [[video](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51936/)]
* [TACO 2025] HEngine: A High Performance Optimization Framework on a GPU for Homomorphic Encryption [[paper](https://dl.acm.org/doi/10.1145/3732942)]
* [arXiv 2025] CAT: A GPU-Accelerated FHE Framework with Its Application to High-Precision Private Dataset Query [[paper](https://cs.paperswithcode.com/paper/cat-a-gpu-accelerated-fhe-framework-with-its)] [[code](https://github.com/Rayman96/CAT/tree/main)]
* [PACT 2024] BoostCom: Towards Efficient Universal Fully Homomorphic Encryption by Boosting the Word-wise Comparisons [[paper](https://dl.acm.org/doi/10.1145/3656019.3676893)]
* [NIPS Safe Generative AI Workshop 2024] Privacy-Preserving Large Language Model Inference via GPU-Accelerated Fully Homomorphic Encryption [[paper](https://openreview.net/forum?id=Rs7h1od6ov)] [[code](https://github.com/leodec/openfhe-gpu-public)]
* [TDSC 2024] Phantom: A CUDA-Accelerated Word-Wise Homomorphic Encryption Library [[paper](https://ieeexplore.ieee.org/document/10428046)] [[code](https://github.com/encryptorion-lab/phantom-fhe)]
* [arXiv 2024] Cheddar: A Swift Fully Homomorphic Encryption Library for CUDA GPUs [[paper](https://arxiv.org/abs/2407.13055)] [[code](https://github.com/scale-snu/cheddar-fhe)]
* [MICRO 2023] GME: Gpu-based microarchitectural extensions to accelerate homomorphic encryption [[paper](https://dl.acm.org/doi/abs/10.1145/3613424.3614279)]
* [HPCA 2023] TensorFHE: Achieving Practical Computation on Encrypted Data Using GPGPU [[paper](https://ieeexplore.ieee.org/document/10071017)] [[code](https://hub.docker.com/r/suen0/tensorfhe)]
* [IPDPS 2023] Towards Faster Fully Homomorphic Encryption Implementation with Integer and Floating-point Computing Power of GPUs [[paper](https://ieeexplore.ieee.org/document/10177431)]
* [TPDS 2023] HE-Booster: An Efficient Polynomial Arithmetic Acceleration on GPUs for Fully Homomorphic Encryption [[paper](https://ieeexplore.ieee.org/document/10012383)]
* [WAHC 2023] GPU Acceleration of High-Precision Homomorphic Computation Utilizing Redundant Representation [[paper](https://dl.acm.org/doi/10.1145/3605759.3625256)] [[code](https://github.com/sh-narisada/cuParmesan)]
* [Access 2023] Homomorphic Encryption on GPU [[paper](https://ieeexplore.ieee.org/document/10097488)] [[code](https://github.com/Alisah-Ozcan/HEonGPU)]
* [TC 2022] CARM: CUDA-Accelerated RNS Multiplication in Word-Wise Homomorphic Encryption Schemes for Internet of Things [[paper](https://dl.acm.org/doi/abs/10.1109/TC.2022.3227874)]
* [IPDPS 2022] Accelerating Encrypted Computing on Intel GPUs [[paper](https://ieeexplore.ieee.org/document/9820676)]
* [TCHES 2021] Over 100x Faster Bootstrapping in Fully Homomorphic Encryption through Memory-centric Optimization with GPUs [[paper](https://tches.iacr.org/index.php/TCHES/article/view/9062)] [[code](https://github.com/scale-snu/ckks-gpu-core)]
* [Access 2021] Accelerating Fully Homomorphic Encryption Through Architecture-Centric Analysis and Optimization [[paper](https://ieeexplore.ieee.org/document/9481143)]
