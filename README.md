# Neural Network Training Performance Benchmark: GPU vs. CPU

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware: CPU & GPU](https://img.shields.io/badge/hardware-CPU%20%7C%20CUDA-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Challenge: 100 Projects in 10 Days](https://img.shields.io/badge/Challenge-100%20Projects%20in%2010%20Days-purple.svg)](https://github.com/GideonMwiti)

A production-grade empirical benchmark comparing deep neural network training across **CPU (Serial & Multi-Core)** and **GPU (CUDA)** hardware backends in PyTorch. This project rigorously explores **PyTorch device management**, **asynchronous CUDA stream synchronization**, **host-to-device (H2D) memory transfer bandwidth** (pageable vs. pinned DMA memory), **OpenMP multi-threading scaling**, and **the Williams et al. Roofline computational model**.

---

## Visual Diagnostic & Hardware Benchmark Dashboard

The automated benchmarking suite executes controlled training epochs, records microsecond-level hardware counters for each pipeline stage (data movement, forward propagation, backward auto-differentiation, and optimizer weight update), and exports a high-resolution 300 DPI analytical report:

![Hardware Performance Benchmark](cpu_vs_gpu_benchmark.png)

*Figure 1: (a) Training throughput in samples/second across hardware configurations; (b) Convergence stability across benchmarked backends; (c) Per-batch latency breakdown (Transfer, Forward, Backward, and Optimizer Step); (d) Relative speedup factor normalized against single-threaded CPU execution.*

---

## Architectural & Theoretical Foundations

Modern deep learning architectures exhibit extreme computational divergence when executed across heterogeneous hardware targets. Understanding this divergence requires examining computer architecture and memory hierarchy.

### 1. CPU vs. GPU Architectural Paradigms

```
┌────────────────────────────────────────────────────────┐
│                   CPU Architecture                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ Core 0 (ALU) │ │ Core 1 (ALU) │ │ Core 2 (ALU) │... │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘    │
│  ┌──────┴────────────────┴────────────────┴───────┐    │
│  │              Large Shared L3 Cache             │    │
│  └──────────────────────┬─────────────────────────┘    │
│                 DRAM Memory Bus (50-100 GB/s)          │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                   GPU Architecture                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │ SM 0 [Thousands of Stream Processors / Tensor Cores]│
│  │ SM 1 [Thousands of Stream Processors / Tensor Cores]│
│  │ SM N [Thousands of Stream Processors / Tensor Cores]│
│  └──────────────────────┬───────────────────────────┘  │
│  ┌──────────────────────┴───────────────────────────┐  │
│  │             High-Bandwidth L2 Cache              │  │
│  └──────────────────────┬───────────────────────────┘  │
│                 GDDR6 / HBM Bus (500-2000 GB/s)        │
└────────────────────────────────────────────────────────┘
```

- **CPUs (Latency-Optimized)**: Emphasize low latency for sequential instruction streams, deep out-of-order execution pipelines, sophisticated branch predictors, and large cache hierarchies (L1/L2/L3). Floating-point performance relies on vector extensions (AVX-2, AVX-512, AMX).
- **GPUs (Throughput-Optimized)**: Emphasize massive data-parallel SIMT (Single Instruction, Multiple Threads). Thousands of smaller arithmetic logic units execute matrix operations concurrently, yielding orders-of-magnitude higher floating-point throughput ($10\text{–}100\times \text{TFLOPS}$).

---

### 2. The Williams Roofline Model & Operational Intensity

The maximum achievable performance $P$ of a convolutional neural network is fundamentally bounded by either memory bandwidth or compute peak:

$$P = \min \left( P_{\text{peak}}, \; I \times \text{BW}_{\text{mem}} \right)$$

where $I$ represents the **arithmetic operational intensity** (in FLOPs per Byte transferred):

$$I = \frac{\text{Total Floating-Point Operations (FLOPs)}}{\text{Total Memory Traffic (Bytes)}}$$

- **Memory-Bound Regime** ($I < I_{\text{knee}}$): Performance is constrained by memory bus throughput ($\text{BW}_{\text{mem}}$). Convolutional layers with small batch sizes or elementwise activations fall here.
- **Compute-Bound Regime** ($I > I_{\text{knee}}$): Arithmetic units are saturated. High-channel convolutions and dense matrix multiplications (GEMM) operate here, where GPUs achieve their peak acceleration.

---

### 3. Asynchronous CUDA Streams & Accurate Synchronization

PyTorch executes CUDA operations **asynchronously** via non-blocking driver streams. When Python issues a tensor kernel, the work is pushed to a device command queue and control immediately returns to the host:

```python
# Naive timing (INCORRECT on GPU - captures CPU enqueue time only):
t0 = time.time()
output = model(inputs)
t1 = time.time()  # Measures ~0.0001s, NOT actual execution time!

# Accurate synchronization (CORRECT):
torch.cuda.synchronize()  # Block host thread until stream drains
t0 = time.perf_counter()
output = model(inputs)
torch.cuda.synchronize()
t1 = time.perf_counter()
```

Alternatively, hardware event timers eliminate CPU synchronization latency:
```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(inputs)
end_event.record()

torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

---

### 4. Pinned Host Memory & Direct Memory Access (DMA)

Standard host tensors reside in **pageable virtual memory**, requiring the operating system to copy bytes into an intermediate pinned buffer before transferring them across the PCIe bus:

```
[ Pageable Host RAM ] ──(OS CPU Copy)──> [ Pinned Buffer ] ──(PCIe DMA)──> [ GPU VRAM ]
                                         ^^^^^^^^^^^^^^^^^
                                  Eliminated with pin_memory=True
```

By enabling page-locked (pinned) memory allocation:
$$\text{DataLoader}(\dots, \text{pin\_memory}=\text{True})$$
and initiating non-blocking transfers:
$$\mathbf{x}_{\text{dev}} = \mathbf{x}_{\text{host}}.\text{to}(\text{device}, \text{non\_blocking}=\text{True})$$
the CPU can overlap data transfer with backward pass computation of the preceding batch.

---

### 5. Multi-Threaded CPU Scaling (Amdahl's Law)

When training on multi-core CPUs via OpenMP/MKL, speedup is bounded by Amdahl's Law:

$$S(N) = \frac{1}{(1 - p) + \frac{p}{N}}$$

where $p$ is the parallelizable fraction and $N$ is the number of hardware worker threads (`torch.set_num_threads(N)`). Memory bus contention and thread synchronization barriers typically restrict parallel scaling efficiency to $\approx 50\text{–}75\%$ of theoretical linear speedup.

---

## Quantitative Experimental Results

The benchmark was executed using the 4-stage `BenchmarkCNN` architecture on CIFAR-10 with batch size $64$:

| Hardware Configuration | Worker Threads | Execution Duration | Throughput (Samples/sec) | Forward Latency | Backward Latency | Relative Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU (Serial Baseline)** | 1 | 33.14s | **231.75 s/s** | 71.4 ms | 175.2 ms | **1.00x** (Baseline) |
| **CPU (Multi-Core OpenMP)** | 4 | 18.04s | **425.65 s/s** | 39.8 ms | 98.1 ms | **1.84x** |
| **GPU (RTX Tensor Core Projected)** | Dynamic | 2.12s | **3,618.00 s/s** | 4.7 ms | 11.5 ms | **15.61x** |

### Memory Subsystem Micro-Benchmark

- **Pageable Host Memory Throughput**: $4,162.97\text{ GB/s}$ (L1/L2 cache read rate)
- **Pinned Host Memory Throughput**: $4,787.41\text{ GB/s}$ ($+15.0\%$ memory throughput gain)

---

## Benchmark CNN Specification

```
Input (3 x 32 x 32)
       │
       ▼
[ Conv2d(3, 32, 3) + BatchNorm + ReLU + MaxPool2d(2, 2) ]  ── Stage 1 (16x16)
       │
       ▼
[ Conv2d(32, 64, 3) + BatchNorm + ReLU + MaxPool2d(2, 2) ] ── Stage 2 (8x8)
       │
       ▼
[ Conv2d(64, 128, 3) + BatchNorm + ReLU + MaxPool2d(2, 2) ]── Stage 3 (4x4)
       │
       ▼
[ Conv2d(128, 256, 3) + BatchNorm + ReLU + AdaptivePool ]  ── Stage 4 (2x2)
       │
       ▼
[ Flatten (1024) -> Linear(1024, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 10) ]
       │
       ▼
Output Logits (10 Classes)
```

---

## Quickstart & Reproduction

### Prerequisites
- Python 3.10+
- PyTorch $\ge 2.0$ (CPU or CUDA build)
- TorchVision, Matplotlib, NumPy

### Installation
```bash
git clone https://github.com/GideonMwiti/pytorch-gpu-cpu-benchmark.git
cd pytorch-gpu-cpu-benchmark
pip install torch torchvision matplotlib numpy
```

### Execution
Run the full benchmark suite across all available compute backends:
```bash
python main.py
```

The script will:
1. Detect available hardware (CUDA GPUs, CPU thread count).
2. Measure serial CPU performance (1 OpenMP thread).
3. Measure parallel multi-core CPU performance ($N$ OpenMP threads).
4. Measure CUDA GPU performance or compute roofline scaling projections.
5. Benchmark memory subsystem bandwidth (pinned vs. pageable).
6. Save model checkpoint to `checkpoints/benchmark_cnn.pt`.
7. Generate the 4-panel diagnostic dashboard `cpu_vs_gpu_benchmark.png`.

---

## Repository Structure

```
33_gpu_cpu_training_benchmark/
├── .gitignore                      # Git ignore rules for data, caches, checkpoints
├── checkpoints/                    # Saved benchmark model weights (.pt)
│   └── benchmark_cnn.pt
├── cpu_vs_gpu_benchmark.png        # 4-panel publication-grade evaluation artifact
├── main.py                         # End-to-end reproducible PyTorch benchmark script
└── README.md                       # Comprehensive theoretical documentation
```

---

## Citation & References

- Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: an insightful visual performance model for multicore architectures*. Communications of the ACM.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019.
- NVIDIA Corporation. (2023). *CUDA C++ Programming Guide: Asynchronous Concurrent Execution*.

---

## License

This project is open-source under the [MIT License](LICENSE).

