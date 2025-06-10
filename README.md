# Privacy-Preserving Machine Learning (PPML) Survey

This repo contains the papers mentioned in paper XXX.

## Protocol-Level Optimization

## Model-Level Optimizations

### Linear Layer Optimizations

* [CCS 2021] COINN: Crypto/ML Codesign for Oblivious Inference via Neural Networks [[paper](https://dl.acm.org/doi/pdf/10.1145/3460120.3484797)] [[code](https://github.com/ACESLabUCSD/COINN)]

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
* [NIPS Safe Generative AI Workshop 2024] Privacy-Preserving Large Language Model Inference via GPU-Accelerated Fully Homomorphic Encryption [[paper](https://openreview.net/pdf?id=Rs7h1od6ov)] [[code](https://github.com/leodec/openfhe-gpu-public)]
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
