# Satellite Image Town Classifier (遥感影像城镇区域分类器)

A project that uses traditional machine learning (SVM) with computer vision techniques (FFT, GLCM) to classify town areas in high-resolution panchromatic satellite imagery.

一个基于C++, OpenCV, GDAL和SVM的遥感影像城镇区域分类项目，旨在实现对高分辨率全色遥感影像中城镇区域的自动识别。

---

## 📖 Table of Contents (目录)

*   [**Project Background (项目背景)**](#-project-background-项目背景)
*   [**Core Technologies (核心技术栈)**](#-core-technologies-核心技术栈)
*   [**Our Journey: From Zero to Hero (我们的旅程：从0到1)**](#-our-journey-from-zero-to-hero-我们的旅程从0到1)
*   [**How to Run (如何运行)**](#-how-to-run-如何运行)
*   [**License (开源协议)**](#-license-开源协议)

---

## 🎯 Project Background (项目背景)

This project was born out of a university research requirement for the selective encryption of large-scale remote sensing images. To enhance the security and efficiency of geospatial data, it's crucial to apply different encryption strategies based on the value of the information. Urban areas, containing dense infrastructure and human activity, are considered high-value targets that require strong encryption, while other areas like farmland or water bodies can be protected with lightweight scrambling.

The primary challenge is to automatically and accurately distinguish between urban and non-urban areas in massive satellite images. This project serves as the "intelligence module" for the subsequent encryption system, providing precise target identification.

该项目诞生于一个关于“超大遥感影像选择性加密”的高校研究课题。为了提升地理空间数据的安全性与处理效率，依据信息价值的不同来实施差异化加密策略至关重要。城镇区域包含了密集的基础设施与人类活动，被视为需要重点加密的高价值目标；而农田、水体等其他区域则可通过轻量级置乱算法进行保护。

本项目的核心挑战在于，如何自动、精准地从海量遥感影像中识别出城镇区域。它作为后续加密系统的“情报识别模块”，为实现精确的目标加密提供了基础。

--

## 🛠️ Core Technologies (核心技术栈)

*   **Programming Language**: C++ (17)
*   **Computer Vision**: OpenCV
*   **Geospatial Data Abstraction**: GDAL
*   **Machine Learning**: Support Vector Machine (SVM) via OpenCV's `ml` module
*   **Feature Extraction**:
    *   **Frequency Domain**: Fast Fourier Transform (FFT) for energy and variance analysis.
    *   **Spatial Domain**: Gray-Level Co-occurrence Matrix (GLCM) for contrast and correlation analysis.
*   **Build System**: CMake

---

## 🚀 Our Journey: From Zero to Hero (我们的旅程：从0到1)

This project was a marathon of problem-solving, a true testament to the spirit of engineering.

1.  **The Initial Crash**: We started with a basic C++ implementation, but it was plagued by countless compilation errors and runtime crashes, especially related to library dependencies (GDAL, OpenCV) and memory management.
2.  **The Debugging Saga**: Through methodical debugging, we conquered linking errors (LNK2019, LNK2005), environment configuration issues, and subtle bugs in matrix operations, like the infamous `cv::Exception` caused by data type mismatches (`CV_32F` vs. `CV_64F`).
3.  **The First Success**: After countless iterations, we finally saw the first red boxes correctly identifying town areas. A milestone moment!
4.  **The Optimization Loop**: Not satisfied with the initial 80-90% accuracy, we delved deeper. We implemented a "Hard Negative Mining" strategy, using a custom-built `SampleSelector` tool to add misclassified samples back into the training set. This significantly improved the model's robustness.
5.  **The Quest for "Why"**: With a working model, we pushed further to understand its decisions. We implemented a feature analysis module to calculate the exact feature ranges ([min, max]) of all identified town areas, moving from a "black box" to an interpretable model.
6.  **The Final Code**: The result is a clean, robust, and well-documented C++ project that not only works but is also a chronicle of our learning and growth.

这个项目是一场解决问题的马拉松，是工程师精神的真实写照。

1.  **最初的崩溃**：我们从一个基础的C++实现开始，但它被无数的编译错误和运行时崩溃所困扰，特别是与库依赖（GDAL, OpenCV）和内存管理相关的部分。
2.  **调试史诗**：通过系统化的调试，我们征服了链接错误（LNK2019, LNK2005）、环境配置问题，以及矩阵运算中的微小bug，比如那个因数据类型不匹配（`CV_32F` vs `CV_64F`）而引发的臭名昭著的`cv::Exception`。
3.  **第一次成功**：经过无数次迭代，我们终于看到了第一个正确框出城镇区域的红色方框。一个里程碑式的时刻！
4.  **优化循环**：不满足于最初80-90%的准确率，我们深入探索。我们实施了“难例挖掘”策略，使用一个自制的`SampleSelector`工具，将被错误分类的样本重新加入训练集，极大地提升了模型的稳健性。
5.  **追问“为什么”**：在模型能工作后，我们进一步探究其决策原理。我们实现了一个特征分析模块，用以计算所有被识别城镇的精确特征范围（[min, max]），将模型从一个“黑箱”变为一个可解释的模型。
6.  **最终的代码**：最终的成果是一个干净、健壮、文档完善的C++项目，它不仅能成功运行，更是我们学习与成长的生动记录。

---

## ⚙️ How to Run (如何运行)

*(This section needs to be filled with your specific environment details, but here is a template)*

*(本节需要您根据您的具体环境进行填充，但以下是一个模板)*

1.  **Prerequisites (环境要求)**
    *   Visual Studio 2022
    *   CMake
    *   OpenCV (e.g., installed at `C:/opencv`)
    *   GDAL (e.g., installed via vcpkg)
    *   A Landsat 8 panchromatic band image (e.g., `LC09_L1TP_124033_20250901_20250901_02_T1_B8.TIF`)

2.  **Configuration (配置)**
    *   Clone this repository.
    *   Update the paths in `CMakeLists.txt` to match your OpenCV and GDAL installation locations.
    *   Update the `filePath` variable in `main.cpp` and `select_samples.cpp` to point to your satellite image.

3.  **Build and Run (编译与运行)**
    *   Open the project folder in Visual Studio.
    *   Wait for CMake to configure the project.
    *   Select `SceneClassifier` or `SampleSelector` as the startup item.
    *   Run the project.

---

## 📄 License (开源协议)

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

本项目基于 MIT 协议开源。详情请见 [LICENSE](LICENSE) 文件。
