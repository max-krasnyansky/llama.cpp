# Agent Instructions: llama.cpp Hexagon Backend

## Overview
You are an AI coding agent tasked with developing, refactoring, and enhancing the Hexagon NPU backend for the `llama.cpp` project. The Hexagon backend accelerates LLM inference by offloading operations from the host CPU (ARM64) to the Qualcomm Hexagon DSP/NPU found in Snapdragon SOCs.

Your primary goals are to ensure strict correctness of tensor operations and to maximize Tokens Per Second (T/S) throughput by writing highly optimized, hardware-aware C/C++ code.

---

## Part 1: Build Process & Toolchain Requirements

When assisting with build scripts, CMake configurations, or deployment tasks, adhere to the following environment constraints and build procedures:

* **Docker-Based Build Environment (Primary):** The standard and recommended method for building `llama.cpp` with the Hexagon backend is using the Snapdragon Toolchain Docker image. When analyzing or generating build steps, assume the environment is orchestrated via the provided `build-snapdragon.sh` script executed within this container.
* **Standard Build Execution:**
  ```bash
  docker run -it -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --platform linux/amd64 \
    ghcr.io/snapdragon-toolchain/arm64-android:v0.6 \
    /workspace/build-snapdragon.sh

```

* **Testing Limitations (Hardware Dependency):** Do **not** attempt to execute test suites or run the compiled Hexagon binaries. Running the backend requires physical access to a Snapdragon-based device (SOC) equipped with a Hexagon NPU. As an AI agent, you do not have access to this hardware. Your scope is strictly limited to generating code, building, cross-compiling, and verifying static correctness rather than runtime execution.
* **Qualcomm Hexagon SDK:** The build relies heavily on the Hexagon SDK. Whether inside the container or assisting with a bare-metal setup, assume standard installation paths. You will frequently need to interact with the SDK's internal toolchains (Clang for Hexagon).
* **Cross-Compilation (Host Code - ARM64):** Manages the `llama.cpp` context, memory allocation, and RPC calls.
* **Cross-Compilation (Device Code - Hexagon v73+):** The actual tensor operation kernels utilizing HVX/HMX.

---

## Part 2: Hexagon NPU Capabilities & Best Practices

When writing or refactoring backend operations (Ops), apply the following architectural guidelines to fully utilize the NPU:

### 1. Compute & HVX Vectorization

* **Vectorized Workloads First:** Use vectorized HVX (Hexagon Vector eXtensions) operations for all compute-heavy workloads.
* **Register Locality:** Avoid mixing scalar code and HVX. Keep values in the HVX registers as long as possible to minimize overhead.
* **Unaligned Element Handling:** If the number of elements in a tensor is not a multiple of the vector length (128/64/32 depending on data type), avoid scalar fallbacks. Instead, use masked operations or partial stores for the output. See examples of `Q6_Q_vsetq_R` and `hvx_vec_store` in `ggml/src/ggml-hexagon/htp/...`

### 2. Operation State & Local Contexts

Avoid recomputing variables on the fly during inference. All Hexagon Ops must be designed to use a local context structure.

* **Precompute & Cache:** Parse Op params and data during the initialization phase precompute and store all static tensor metadata, strides, and quantization parameters in the Op's local context.
* **Execution Loop:** The inference execution loop should only read from this cached context to maximize prompt processing speed.

### 3. Memory Hierarchy & Bandwidth

Memory bound operations are the primary bottleneck for LLMs. Strict memory management is required:

* **VTCM & DMA Routing:** For large data chunks (16KB and up), always use asynchronous DMA to transfer data from DDR into VTCM (Vector Tightly Coupled Memory), and then use HVX/HMX for the compute.
* **Pipelining (Double Buffering):** Use double (or multiple) buffering in VTCM to pipeline operations. Always aim to have outstanding DMA transactions running while simultaneously processing a chunk of data with HVX or HMX. See examples in `ggml/src/ggml-hexagon/htp/matmul-ops.c`.
* **2D DMA Alignment:** Utilize the DMA engine's 2D mode to properly align the data in memory for HVX (e.g., ensuring 128-byte alignment).
* **Strictly Avoid Scalar VTCM Access:** Scalar access to VTCM generates L2 cache misses. Use HVX for VTCM read/write operations.
* **VTCM Population Exception:** It is acceptable to use scalar code to populate small chunks of VTCM with initial values. Utilize the helper functions in `hvx-utils` for value replication, sum reduction, etc.

### 4. Scaling & Threading

Snapdragon hardware environments are highly concurrent.

* **Hardware Thread Counts:** The Hexagon architecture typically features 4-8 HVX threads and 6-10 total HW threads.
* **Parallel DMA Engines:** Each thread is equipped with its own DMA engine that can run in parallel with the other hardware threads. Ensure memory operations take advantage of this concurrent structure.
* **Thread Groups:** When implementing dispatch logic, utilize the `ngrp` attribute to explicitly manage the total number of thread groups (mapping one group per available NPU). Ensure workloads are evenly sharded to prevent bottlenecking.

---

## Part 3: Official Documentation & References

When implementing specific instructions, verifying register behavior, or optimizing HMX/HVX pipelines, consult the following official Qualcomm Hexagon Programmer's Reference Manuals:

* **Hexagon v73 Base Architecture:** [80-N2040-53](https://docs.qualcomm.com/doc/80-N2040-53/80-N2040-53.pdf)
* **Hexagon v73 HVX Reference:** [80-N2040-54_REV_AB](https://docs.qualcomm.com/doc/80-N2040-54/80-N2040-54_REV_AB_Qualcomm_Hexagon_V73_HVX_Programmers_Reference_Manual.pdf)
* **Hexagon v79 Architecture Topic:** [80-N2040-60](https://docs.qualcomm.com/doc/80-N2040-60/topic)
* **Hexagon v79 HVX Topic:** [80-N2040-61](https://docs.qualcomm.com/doc/80-N2040-61/topic)
* **Hexagon v81 HMX Reference (Applicable to v73 - v81):** [80-N2040-62_REV_AA](https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf)
