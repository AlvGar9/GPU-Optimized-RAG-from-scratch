<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README</title>
</head>
<body>

<div align="center">
  <h1>GPU-Accelerated RAG from scratch and Model Serving System.</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NVIDIA%20CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/CuPy-23A459?style=for-the-badge" alt="CuPy">
  <img src="https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
</div>

<br>

<div align="center">
  <img src="assets/Autoscaler_Flow_Diag.png" width="350"/>
</div>

## 1. Project Overview

[cite_start]This project presents a high-performance system for information retrieval, designed for large-scale, high-dimensional vector search[cite: 59]. It integrates GPU-level kernel optimizations with a robust, scalable model serving architecture. The system features two primary components:

1.  [cite_start]A **GPU-accelerated search pipeline** for both K-Nearest Neighbors (KNN) and Approximate Nearest Neighbors (ANN) that demonstrates significant speedups over traditional CPU-based methods[cite: 61].
2.  [cite_start]A **FastAPI-based serving system** for Retrieval-Augmented Generation (RAG) that handles real-time inference under high load using request queuing, batching, and dynamic autoscaling[cite: 62, 497].

This repository provides the source code for implementing, benchmarking, and serving these advanced machine learning systems.

## 2. Motivation

### Why GPU Acceleration?
[cite_start]Modern machine learning applications in semantic search, recommendation, and classification depend on finding the most relevant items from vast datasets[cite: 66]. This is typically framed as a nearest neighbor search over high-dimensional vectors. [cite_start]As datasets grow in size ($N$) and dimensionality ($D$), exact search becomes a computational bottleneck, with CPU throughput and I/O being primary constraints[cite: 68, 74].

[cite_start]GPUs, with their massively parallel architecture, are designed for the high-throughput, repetitive operations inherent in distance calculations[cite: 77]. [cite_start]By offloading this work to the GPU, we can transform response times from seconds to milliseconds, enabling low-latency retrieval that is crucial for production applications[cite: 78, 539].

### Why Advanced Model Serving?
Deploying complex models like RAG systems introduces challenges distinct from standard web services. [cite_start]These models demand high throughput and low latency, and their computational intensity can easily overload a server during traffic spikes[cite: 374, 375].

To address this, system-level optimizations are essential. [cite_start]Techniques like request queuing and batching smooth out traffic, maximize GPU utilization, and create a balance between throughput and latency[cite: 377, 380]. [cite_start]Furthermore, an autoscaling mechanism ensures that computational resources dynamically match demand, preventing both resource underutilization during quiet periods and performance degradation under high load[cite: 1118, 1119].

## 3. System Architecture and Implementation

### How It Works: GPU-Accelerated Search

The core of the retrieval pipeline is built on efficient, parallelized distance computations.

* [cite_start]**Distance Functions**: The system implements and benchmarks four fundamental distance metrics: **L2 (Euclidean)**, **Dot Product**, **Cosine Similarity**, and **Manhattan (L1)**[cite: 153]. These functions are implemented across multiple frameworks to compare performance, including:
    * **NumPy** as a CPU baseline.
    * **PyTorch** and **CuPy** for high-level GPU acceleration.
    * [cite_start]**Custom CUDA Kernels** and **Triton** for low-level control and maximum performance, utilizing optimizations like shared memory and parallel reduction[cite: 60, 164, 366, 367].

* **Retrieval Pipelines**:
    * **Exact KNN**: A brute-force implementation that compares a query vector against all dataset vectors. [cite_start]The GPU version processes the dataset in chunks to manage memory and uses a hybrid top-k selection strategy (partial sort on GPU, final sort on CPU)[cite: 282, 284].
    * [cite_start]**Approximate ANN (LSH)**: To accelerate search, a Locality-Sensitive Hashing (LSH) based ANN system is implemented[cite: 289]. [cite_start]It uses a two-stage process: first, it rapidly identifies a small set of candidate vectors using a Hamming distance prefilter, and second, it performs exact distance computations on only these candidates using the optimized GPU kernels[cite: 82, 83].

### How It Works: Scalable RAG Serving System

The serving system is designed for resilience and performance under real-world conditions.

* [cite_start]**Request Queue and Batcher**: Incoming requests are managed by a thread-safe queue, which decouples the API endpoint from the processing logic[cite: 452, 454]. [cite_start]A background worker assembles requests into batches based on either size (`MAX_BATCH_SIZE = 8`) or a timeout (`MAX_WAITING_TIME = 0.1s`), ensuring a balance between latency and computational efficiency[cite: 455].

* [cite_start]**Autoscaler Architecture**: The system features a distributed architecture to handle variable workloads dynamically[cite: 1124].
    * [cite_start]A **Head Node** acts as the entry point, receiving all client queries and placing them in a central queue[cite: 1125, 1126].
    * An **Autoscaler** component monitors the queue length every 15 seconds. [cite_start]It scales the number of **Compute Nodes** up or down based on traffic[cite: 498, 1127].
    * [cite_start]A **Request Dispatcher** uses a round-robin algorithm to distribute tasks evenly across the available compute nodes, ensuring balanced workloads[cite: 1129].

## 4. Project Structure

The repository is organized to separate concerns, with distinct modules for data handling, modeling, evaluation, and testing.

-   `src/data_ingestion/`: Contains modules for loading data.
-   `src/data_processing/`:
    -   `distances.py`: Implementation of the L2, Dot, Cosine, and Manhattan distance functions across different frameworks.
-   `src/evaluation/`:
    -   `benchmark.py`: Scripts for performance testing and benchmarking of the distance functions and retrieval pipelines.
-   `src/modeling/`:
    -   `knn.py`: Core logic for the KNN and ANN (LSH-based) retrieval algorithms.
    -   `rag.py`: Implementation of the RAG pipeline and the FastAPI serving system, including the request queue and batcher.
-   `src/tests/`:
    -   Unit tests for the distance functions, KNN/ANN algorithms, and RAG components to ensure correctness and stability.

</body>
</html>
