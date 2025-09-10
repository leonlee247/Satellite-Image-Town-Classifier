#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <chrono>
#include <map>
#include <limits> // 为了使用 std::numeric_limits

#pragma warning(push, 0) 
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"
#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#pragma warning(pop) 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- 特征结构体 ---
struct BlockFeatures {
    cv::Rect position;
    double fft_energy = 0.0;
    double fft_variance = 0.0;
    double glcm_contrast = 0.0;
    double glcm_correlation = 0.0;
};

// --- GLCM 计算 ---
cv::Mat calculateGLCM(const cv::Mat& block, int dx, int dy) {
    cv::Mat glcm = cv::Mat::zeros(256, 256, CV_64F);
    for (int y = 0; y < block.rows - dy; ++y) {
        for (int x = 0; x < block.cols - dx; ++x) {
            glcm.at<double>(block.at<uchar>(y, x), block.at<uchar>(y + dy, x + dx)) += 1.0;
        }
    }
    cv::Scalar sum = cv::sum(glcm);
    if (sum[0] > 0) glcm /= sum[0];
    return glcm;
}
double calculateGLCMContrast(const cv::Mat& glcm) {
    double contrast = 0.0;
    for (int i = 0; i < 256; ++i) for (int j = 0; j < 256; ++j) contrast += (i - j) * (i - j) * glcm.at<double>(i, j);
    return contrast;
}
double calculateGLCMCorrelation(const cv::Mat& glcm) {
    double mean_i = 0, mean_j = 0, std_i = 0, std_j = 0, correlation = 0;
    cv::Mat p_x = cv::Mat::zeros(1, 256, CV_64F), p_y = cv::Mat::zeros(1, 256, CV_64F);
    for (int i = 0; i < 256; ++i) for (int j = 0; j < 256; ++j) { p_x.at<double>(i) += glcm.at<double>(i, j); p_y.at<double>(j) += glcm.at<double>(i, j); }
    for (int i = 0; i < 256; ++i) { mean_i += i * p_x.at<double>(i); mean_j += i * p_y.at<double>(i); }
    for (int i = 0; i < 256; ++i) { std_i += (i - mean_i) * (i - mean_i) * p_x.at<double>(i); std_j += (i - mean_j) * (i - mean_j) * p_y.at<double>(i); }
    std_i = sqrt(std_i); std_j = sqrt(std_j);
    if (std_i < 1e-6 || std_j < 1e-6) return 0;
    for (int i = 0; i < 256; ++i) for (int j = 0; j < 256; ++j) correlation += (i - mean_i) * (j - mean_j) * glcm.at<double>(i, j) / (std_i * std_j);
    return correlation;
}

// --- 其他辅助函数 ---
std::vector<cv::Mat> create_directional_masks(int rows, int cols, int num_directions) {
    std::vector<cv::Mat> masks;
    int center_r = rows / 2; int center_c = cols / 2; double angle_step = 180.0 / num_directions;
    for (int i = 0; i < num_directions; ++i) {
        double start_angle = i * angle_step, end_angle = (i + 1) * angle_step;
        cv::Mat mask = cv::Mat::zeros(rows, cols, CV_64F);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
            double angle_rad = std::atan2(static_cast<double>(r - center_r), static_cast<double>(c - center_c));
            double angle_deg = angle_rad * 180.0 / M_PI; if (angle_deg < 0) angle_deg += 180.0;
            if ((angle_deg >= start_angle && angle_deg < end_angle) || (angle_deg + 180.0 >= start_angle && angle_deg + 180.0 < end_angle)) {
                mask.at<double>(r, c) = 1.0;
            }
        }
        masks.push_back(mask);
    }
    return masks;
}
cv::Mat normalize_tile(const cv::Mat& src_tile, double min_val, double max_val) {
    cv::Mat dst; src_tile.convertTo(dst, CV_64F);
    if (std::abs(max_val - min_val) < 1e-6) { dst = cv::Mat::zeros(src_tile.size(), CV_64F); }
    else { dst = (dst - min_val) * (255.0 / (max_val - min_val)); }
    cv::threshold(dst, dst, 255.0, 255.0, cv::THRESH_TRUNC); cv::threshold(dst, dst, 0.0, 0.0, cv::THRESH_TOZERO);
    dst.convertTo(dst, CV_8U);
    return dst;
}

// --- 特征提取 ---
BlockFeatures extract_features_for_block(const cv::Mat& image_8u, const cv::Point& block_tl, int block_size, const std::vector<cv::Mat>& dir_masks) {
    BlockFeatures f;
    f.position = cv::Rect(block_tl.x, block_tl.y, block_size, block_size);
    if (block_tl.x + block_size > image_8u.cols || block_tl.y + block_size > image_8u.rows) {
        return f;
    }
    cv::Mat block_8u = image_8u(f.position);

    if (cv::sum(block_8u)[0] < 1e-6) return f;

    cv::Mat block_64f; block_8u.convertTo(block_64f, CV_64F);
    cv::Mat complexI; cv::merge(std::vector<cv::Mat>{block_64f, cv::Mat::zeros(block_64f.size(), CV_64F)}, complexI);
    cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat planes[2]; cv::split(complexI, planes);
    cv::Mat magI; cv::magnitude(planes[0], planes[1], magI);
    int cx = magI.cols / 2; int cy = magI.rows / 2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy)), q1(magI, cv::Rect(cx, 0, cx, cy)), q2(magI, cv::Rect(0, cy, cx, cy)), q3(magI, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp; q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3); q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    magI += cv::Scalar::all(1); cv::log(magI, magI);
    std::vector<double> directional_energies; double total_energy = 0;
    int num_directions = dir_masks.size();
    for (int j = 0; j < num_directions; ++j) {
        double dir_energy = cv::sum(magI.mul(dir_masks[j]))[0];
        directional_energies.push_back(dir_energy); total_energy += dir_energy;
    }
    f.fft_energy = total_energy;
    double mean_energy = total_energy / num_directions; double variance = 0;
    for (double energy : directional_energies) { variance += std::pow(energy - mean_energy, 2); }
    f.fft_variance = variance / num_directions;

    cv::Mat glcm = calculateGLCM(block_8u, 1, 0);
    f.glcm_contrast = calculateGLCMContrast(glcm);
    f.glcm_correlation = calculateGLCMCorrelation(glcm);

    return f;
}

int main(int argc, char* argv[]) {
    std::string filePath = "D:\\BaiduNetdiskDownload\\LC09_L1TP_124033_20250901_20250901_02_T1_B8.TIF";

    std::cout << "--- Machine Learning Scene Classifier (v3.0 - with Feature Analysis) ---" << std::endl;

    // --- Step 1 & 2: 读取图像 ---
    std::cout << "[Step 1] Reading image..." << std::endl;
    GDALAllRegister();
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(filePath.c_str(), GA_ReadOnly);
    if (poDataset == nullptr) { std::cerr << "Error opening file." << std::endl; return -1; }
    int width = poDataset->GetRasterXSize(); int height = poDataset->GetRasterYSize();
    std::cout << "[Step 2] Image dimensions: " << width << "x" << height << "." << std::endl;

    // --- Step 3: 生成8位灰度图 ---
    std::cout << "[Step 3] Generating 8-bit base image..." << std::endl;
    cv::Mat display_image_8u = cv::Mat::zeros(height, width, CV_8U);
    GDALRasterBand* poBand = poDataset->GetRasterBand(1);
    unsigned int seed = 1234;
    std::mt19937 gen(seed); std::uniform_int_distribution<> row_dist(0, height - 1); std::uniform_int_distribution<> col_dist(0, width - 1);
    std::vector<ushort> samples; samples.reserve(200000);
    for (int i = 0; i < 200000; ++i) { int r = row_dist(gen); int c = col_dist(gen); ushort p; poBand->RasterIO(GF_Read, c, r, 1, 1, &p, 1, 1, GDT_UInt16, 0, 0); if (p > 0) samples.push_back(p); }
    if (samples.empty()) { std::cerr << "No valid pixels found." << std::endl; return -1; } std::sort(samples.begin(), samples.end());
    double global_min_val = samples[static_cast<size_t>(samples.size() * 0.02)]; double global_max_val = samples[static_cast<size_t>(samples.size() * 0.98)];
    int tile_size = 1024; cv::Mat tile_16u_buffer(tile_size, tile_size, CV_16U);
    for (int y = 0; y < height; y += tile_size) for (int x = 0; x < width; x += tile_size) {
        int cw = std::min(tile_size, width - x); int ch = std::min(tile_size, height - y);
        poBand->RasterIO(GF_Read, x, y, cw, ch, tile_16u_buffer.data, cw, ch, GDT_UInt16, 0, 0);
        cv::Mat roi = tile_16u_buffer(cv::Rect(0, 0, cw, ch));
        normalize_tile(roi, global_min_val, global_max_val).copyTo(display_image_8u(cv::Rect(x, y, cw, ch)));
    }

    const int block_size = 256;
    int num_directions = 4;
    std::vector<cv::Mat> dir_masks = create_directional_masks(block_size, block_size, num_directions);

    // --- Step 4: 提取所有区块的特征 ---
    std::cout << "[Step 4] Extracting features for all blocks..." << std::endl;
    std::vector<BlockFeatures> all_features;
    for (int r = 0; r <= height - block_size; r += block_size) {
        for (int c = 0; c <= width - block_size; c += block_size) {
            all_features.push_back(extract_features_for_block(display_image_8u, cv::Point(c, r), block_size, dir_masks));
        }
    }
    std::cout << "  - Feature extraction complete. Total blocks: " << all_features.size() << std::endl;

    // --- Step 5: 准备训练数据 ---
    std::cout << "[Step 5] Preparing training data..." << std::endl;
    std::vector<cv::Point> positive_samples = {
        cv::Point(12288, 8192), cv::Point(5120, 3840), cv::Point(9216, 10752),
        cv::Point(12288, 12800), cv::Point(10496, 15104), cv::Point(7680, 4352),
        cv::Point(12288, 6912), cv::Point(12288, 13056), cv::Point(6144, 13824),
        cv::Point(9216, 10752), cv::Point(12288, 9984)
    };
    std::vector<cv::Point> negative_samples = {
        cv::Point(7936, 8960), cv::Point(3584, 11776), cv::Point(8192, 7168),
        cv::Point(2816, 10496), cv::Point(1792, 7424), cv::Point(2304, 7424),
        cv::Point(9984, 2048), cv::Point(12032, 2816), cv::Point(8448, 4096),
        cv::Point(3584, 256), cv::Point(3584, 512), cv::Point(4608, 512),
        cv::Point(2816, 7680), cv::Point(3584, 9216), cv::Point(7680, 3584),
        cv::Point(7680, 2048), cv::Point(10496, 2560), cv::Point(1792, 2816),
        cv::Point(6144, 4352), cv::Point(1280, 9728)
    };
    int num_samples = positive_samples.size() + negative_samples.size();
    int num_features = 4;
    cv::Mat training_data(num_samples, num_features, CV_32F);
    cv::Mat labels(num_samples, 1, CV_32S);
    int sample_idx = 0;
    for (const auto& p : positive_samples) {
        BlockFeatures f = extract_features_for_block(display_image_8u, p, block_size, dir_masks);
        training_data.at<float>(sample_idx, 0) = f.fft_energy;
        training_data.at<float>(sample_idx, 1) = f.fft_variance;
        training_data.at<float>(sample_idx, 2) = f.glcm_contrast;
        training_data.at<float>(sample_idx, 3) = f.glcm_correlation;
        labels.at<int>(sample_idx, 0) = 1;
        sample_idx++;
    }
    for (const auto& p : negative_samples) {
        BlockFeatures f = extract_features_for_block(display_image_8u, p, block_size, dir_masks);
        training_data.at<float>(sample_idx, 0) = f.fft_energy;
        training_data.at<float>(sample_idx, 1) = f.fft_variance;
        training_data.at<float>(sample_idx, 2) = f.glcm_contrast;
        training_data.at<float>(sample_idx, 3) = f.glcm_correlation;
        labels.at<int>(sample_idx, 0) = -1;
        sample_idx++;
    }
    cv::Mat mean_64f, stddev_64f;
    cv::reduce(training_data, mean_64f, 0, cv::REDUCE_AVG, CV_64F);
    cv::Mat mean_32f;
    mean_64f.convertTo(mean_32f, CV_32F);
    cv::Mat sq_diff;
    cv::pow(training_data - cv::repeat(mean_32f, training_data.rows, 1), 2, sq_diff);
    cv::reduce(sq_diff, stddev_64f, 0, cv::REDUCE_AVG, CV_64F);
    cv::sqrt(stddev_64f, stddev_64f);
    for (int i = 0; i < training_data.rows; ++i) {
        for (int j = 0; j < training_data.cols; ++j) {
            if (stddev_64f.at<double>(0, j) > 1e-6) {
                training_data.at<float>(i, j) = (training_data.at<float>(i, j) - mean_64f.at<double>(0, j)) / stddev_64f.at<double>(0, j);
            }
        }
    }
    std::cout << "  - Feature scaling complete for " << num_samples << " samples." << std::endl;

    // --- Step 6: 训练SVM分类器 ---
    std::cout << "[Step 6] Training the SVM classifier..." << std::endl;
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->trainAuto(cv::ml::TrainData::create(training_data, cv::ml::ROW_SAMPLE, labels));
    std::cout << "  - SVM training complete. Optimal Gamma: " << svm->getGamma() << ", Optimal C: " << svm->getC() << std::endl;

    // --- Step 7: 预测并分析特征 ---
    std::cout << "[Step 7] Predicting all blocks and analyzing features..." << std::endl;
    cv::Mat display_image_bgr;
    cv::cvtColor(display_image_8u, display_image_bgr, cv::COLOR_GRAY2BGR);
    int town_count = 0;

    double min_energy = std::numeric_limits<double>::max(), max_energy = -std::numeric_limits<double>::max();
    double min_variance = std::numeric_limits<double>::max(), max_variance = -std::numeric_limits<double>::max();
    double min_contrast = std::numeric_limits<double>::max(), max_contrast = -std::numeric_limits<double>::max();
    double min_correlation = std::numeric_limits<double>::max(), max_correlation = -std::numeric_limits<double>::max();

    for (const auto& f : all_features) {
        if (f.fft_energy == 0) continue;
        cv::Mat sample(1, num_features, CV_32F);
        sample.at<float>(0, 0) = f.fft_energy;
        sample.at<float>(0, 1) = f.fft_variance;
        sample.at<float>(0, 2) = f.glcm_contrast;
        sample.at<float>(0, 3) = f.glcm_correlation;

        for (int j = 0; j < sample.cols; ++j) {
            if (stddev_64f.at<double>(0, j) > 1e-6) {
                sample.at<float>(0, j) = (sample.at<float>(0, j) - mean_64f.at<double>(0, j)) / stddev_64f.at<double>(0, j);
            }
        }

        if (svm->predict(sample) == 1) {
            cv::rectangle(display_image_bgr, f.position, cv::Scalar(0, 0, 255), 3);
            town_count++;

            min_energy = std::min(min_energy, f.fft_energy);
            max_energy = std::max(max_energy, f.fft_energy);

            min_variance = std::min(min_variance, f.fft_variance);
            max_variance = std::max(max_variance, f.fft_variance);

            min_contrast = std::min(min_contrast, f.glcm_contrast);
            max_contrast = std::max(max_contrast, f.glcm_contrast);

            min_correlation = std::min(min_correlation, f.glcm_correlation);
            max_correlation = std::max(max_correlation, f.glcm_correlation);
        }
    }
    std::cout << "  - Total towns identified: " << town_count << std::endl;

    std::cout << "\n--- Feature Analysis Report for Identified Towns ---" << std::endl;
    if (town_count > 0) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "1. FFT Energy       : [" << min_energy << ", " << max_energy << "]" << std::endl;
        std::cout << "2. FFT Variance     : [" << min_variance << ", " << max_variance << "]" << std::endl;
        std::cout << "3. GLCM Contrast    : [" << min_contrast << ", " << max_contrast << "]" << std::endl;
        std::cout << "4. GLCM Correlation : [" << min_correlation << ", " << max_correlation << "]" << std::endl;
    }
    else {
        std::cout << "No towns were identified, feature analysis is not available." << std::endl;
    }
    std::cout << "----------------------------------------------------" << std::endl;

    // --- Step 8: 保存并显示结果 ---
    for (const auto& p : positive_samples) cv::rectangle(display_image_bgr, cv::Rect(p.x, p.y, block_size, block_size), cv::Scalar(0, 255, 0), 3);
    for (const auto& p : negative_samples) cv::rectangle(display_image_bgr, cv::Rect(p.x, p.y, block_size, block_size), cv::Scalar(255, 0, 0), 3);

    std::string resultImagePath = "town_detection_result_v3.0.png";
    cv::imwrite(resultImagePath, display_image_bgr);
    std::cout << "\n[Step 8] Final result saved to: " << resultImagePath << std::endl;

    cv::Mat display_small;
    double final_scale = std::min(1200.0 / width, 800.0 / height);
    cv::resize(display_image_bgr, display_small, cv::Size(), final_scale, final_scale);
    cv::imshow("Final Classification Result", display_small);

    GDALClose(poDataset);
    std::cout << "\nProcessing complete. Press any key to exit." << std::endl;
    cv::waitKey(0);

    return 0;
}