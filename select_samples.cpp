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

// 使用#pragma来抑制来自第三方库的内部警告
#pragma warning(push, 0) 
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#pragma warning(pop) 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- 全局变量，用于鼠标回调 ---
cv::Point g_mouse_position;
double g_scale = 1.0;

// --- 鼠标回调函数 ---
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_MOUSEMOVE) {
        g_mouse_position.x = x;
        g_mouse_position.y = y;
    }
}

// 特征结构体
struct BlockFeatures {
    cv::Rect position;
    double fft_energy = 0.0;
    double fft_variance = 0.0;
    double glcm_contrast = 0.0;
    double glcm_correlation = 0.0;
};

// GLCM 计算函数
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

// 其他辅助函数
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

int main(int argc, char* argv[]) {
    // -------------------------------------------------------------------
    // --- 模式开关：设置为 true 以拾取坐标；设置为 false 以运行网格搜索 ---
    const bool coordinate_picking_mode = true;
    // -------------------------------------------------------------------

    // --- 更新为河北影像路径 ---
    std::string filePath = "D:\\BaiduNetdiskDownload\\LC09_L1TP_124033_20250901_20250901_02_T1_B8.TIF";

    std::cout << "--- Intelligent Scene Classifier with Grid Search ---" << std::endl;
    std::cout << "[Step 1] Reading image: " << filePath << std::endl;
    GDALAllRegister();
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(filePath.c_str(), GA_ReadOnly);
    if (poDataset == nullptr) { std::cerr << "Error opening file." << std::endl; return -1; }
    int width = poDataset->GetRasterXSize(); int height = poDataset->GetRasterYSize();
    std::cout << "[Step 2] Image dimensions: " << width << "x" << height << "." << std::endl;

    std::cout << "[Step 3-6] Generating 8-bit base image for analysis..." << std::endl;
    cv::Mat display_image_8u = cv::Mat::zeros(height, width, CV_8U);
    GDALRasterBand* poBand = poDataset->GetRasterBand(1);
    unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
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
    cv::Mat display_image_bgr, display_small;

    if (coordinate_picking_mode) {
        std::cout << "\n--- COORDINATE PICKING MODE (HEBEI) ---" << std::endl;
        std::cout << "1. Please select new coordinates for towns and farmlands." << std::endl;
        std::cout << "2. Fill them into 'positive_samples' and 'negative_samples'." << std::endl;
        std::cout << "3. Set 'coordinate_picking_mode' to 'false' and re-run for victory!" << std::endl;

        cv::cvtColor(display_image_8u, display_image_bgr, cv::COLOR_GRAY2BGR);
        for (int y = 0; y < display_image_bgr.rows; y += block_size) cv::line(display_image_bgr, cv::Point(0, y), cv::Point(display_image_bgr.cols, y), cv::Scalar(50, 50, 50), 1);
        for (int x = 0; x < display_image_bgr.cols; x += block_size) cv::line(display_image_bgr, cv::Point(x, 0), cv::Point(x, display_image_bgr.rows), cv::Scalar(50, 50, 50), 1);

        g_scale = std::min(1200.0 / width, 800.0 / height);
        cv::resize(display_image_bgr, display_small, cv::Size(), g_scale, g_scale);

        std::string window_name = "Coordinate Picker";
        cv::namedWindow(window_name);
        cv::setMouseCallback(window_name, onMouse, NULL);

        while (true) {
            cv::Mat frame = display_small.clone();
            int real_x = static_cast<int>(g_mouse_position.x / g_scale);
            int real_y = static_cast<int>(g_mouse_position.y / g_scale);
            std::string coord_text = "Original Img Coords: (" + std::to_string(real_x) + ", " + std::to_string(real_y) + ")";
            cv::rectangle(frame, cv::Point(0, frame.rows - 20), cv::Point(350, frame.rows), cv::Scalar(0, 0, 0), -1);
            cv::putText(frame, coord_text, cv::Point(10, frame.rows - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            cv::imshow(window_name, frame);

            if (cv::waitKey(20) >= 0) {
                break;
            }
        }
        cv::destroyAllWindows();
        GDALClose(poDataset);
        return 0;
    }

    // --- 智能分析模式 ---
    std::cout << "[Step 7] Starting feature extraction for all blocks..." << std::endl;
    std::vector<BlockFeatures> all_features;
    cv::Rect valid_rect = cv::boundingRect(display_image_8u);
    int num_directions = 4;
    std::vector<cv::Mat> dir_masks = create_directional_masks(block_size, block_size, num_directions);

    for (int r = valid_rect.y; r <= valid_rect.y + valid_rect.height - block_size; r += block_size) {
        for (int c = valid_rect.x; c <= valid_rect.x + valid_rect.width - block_size; c += block_size) {
            cv::Mat block_8u = display_image_8u(cv::Rect(c, r, block_size, block_size));
            if (cv::sum(block_8u)[0] > 1e-6) {
                BlockFeatures f;
                f.position = cv::Rect(c, r, block_size, block_size);

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

                all_features.push_back(f);
            }
        }
    }
    std::cout << "[Step 8] Feature extraction complete. Total blocks: " << all_features.size() << std::endl;

    std::cout << "[Step 9] Reading ground truth annotations..." << std::endl;
    // --- 请在这里填入您在河北影像上新选取的坐标 ---
    std::vector<cv::Point> positive_samples = {
        // 示例: cv::Point(x, y),
    };
    std::vector<cv::Point> negative_samples = {
        // 示例: cv::Point(x, y),
    };
    if (positive_samples.empty() || negative_samples.empty()) {
        std::cerr << "Error: Annotation samples are empty. Please run in coordinate_picking_mode first." << std::endl;
        GDALClose(poDataset);
        return -1;
    }
    std::cout << "  - Positive samples (Towns): " << positive_samples.size() << std::endl;
    std::cout << "  - Negative samples (Non-Towns): " << negative_samples.size() << std::endl;

    std::cout << "[Step 10] Starting Grid Search for best parameters..." << std::endl;
    // -------------------------------------------------------------------
    // --- 华北平原专用参数网格 (North China Plain-Specific Grid) ---
    // -------------------------------------------------------------------
    std::map<std::string, std::vector<double>> param_grid = {
        // 提高能量门槛，因为平原地区的城镇能量更高
        {"energy", {30000.0, 35000.0, 40000.0, 45000.0}},
        // 方差阈值范围可以适当收紧
        {"variance", {200000.0, 400000.0, 600000.0}},
        // 大幅提高对比度门槛，这是区分平原城镇和农田的关键
        {"contrast", {20.0, 25.0, 30.0, 35.0}},
        // 严格限制相关性上限，以有效过滤掉纹理规则的农田
        {"correlation", {0.7, 0.75, 0.8, 0.85}}
    };
    // -------------------------------------------------------------------

    double best_f1_score = -1.0;
    std::map<std::string, double> best_params;
    int total_combinations = 1;
    for (auto const& [key, val] : param_grid) total_combinations *= val.size();
    int current_combination = 0;

    for (double energy_t : param_grid["energy"]) {
        for (double variance_t : param_grid["variance"]) {
            for (double contrast_t : param_grid["contrast"]) {
                for (double correlation_t : param_grid["correlation"]) {
                    current_combination++;
                    std::cout << "\r  - Testing combination " << current_combination << "/" << total_combinations << "..." << std::flush;

                    int true_positives = 0, false_positives = 0, false_negatives = 0;

                    for (const auto& f : all_features) {
                        bool is_classified_as_town = (f.fft_energy > energy_t && f.fft_variance < variance_t && f.glcm_contrast > contrast_t && f.glcm_correlation < correlation_t);
                        bool is_true_positive = std::find(positive_samples.begin(), positive_samples.end(), f.position.tl()) != positive_samples.end();
                        bool is_true_negative = std::find(negative_samples.begin(), negative_samples.end(), f.position.tl()) != negative_samples.end();

                        if (is_classified_as_town && is_true_positive) true_positives++;
                        if (is_classified_as_town && is_true_negative) false_positives++;
                        if (!is_classified_as_town && is_true_positive) false_negatives++;
                    }

                    double precision = (true_positives + false_positives > 0) ? (double)true_positives / (true_positives + false_positives) : 0;
                    double recall = (true_positives + false_negatives > 0) ? (double)true_positives / (true_positives + false_negatives) : 0;
                    double f1_score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

                    if (f1_score > best_f1_score) {
                        best_f1_score = f1_score;
                        best_params = { {"energy", energy_t}, {"variance", variance_t}, {"contrast", contrast_t}, {"correlation", correlation_t} };
                    }
                }
            }
        }
    }
    std::cout << "\n[Step 11] Grid Search complete!" << std::endl;

    std::cout << "\n--- Optimal Parameters Found ---" << std::endl;
    std::cout << "Best F1-Score: " << best_f1_score << std::endl;
    std::cout << "  - energy_threshold > " << best_params["energy"] << std::endl;
    std::cout << "  - variance_threshold < " << best_params["variance"] << std::endl;
    std::cout << "  - contrast_threshold > " << best_params["contrast"] << std::endl;
    std::cout << "  - correlation_threshold < " << best_params["correlation"] << std::endl;

    std::cout << "\n[Step 12] Drawing final result with optimal parameters..." << std::endl;
    cv::cvtColor(display_image_8u, display_image_bgr, cv::COLOR_GRAY2BGR);
    int town_count = 0;
    for (const auto& f : all_features) {
        if (f.fft_energy > best_params["energy"] && f.fft_variance < best_params["variance"] && f.glcm_contrast > best_params["contrast"] && f.glcm_correlation < best_params["correlation"]) {
            cv::rectangle(display_image_bgr, f.position, cv::Scalar(0, 0, 255), 3); // 红色: 识别出的城镇
            town_count++;
        }
    }
    for (const auto& p : positive_samples) cv::rectangle(display_image_bgr, cv::Rect(p.x, p.y, block_size, block_size), cv::Scalar(0, 255, 0), 3); // 绿色: 真城镇样本
    for (const auto& p : negative_samples) cv::rectangle(display_image_bgr, cv::Rect(p.x, p.y, block_size, block_size), cv::Scalar(255, 0, 0), 3); // 蓝色: 真非城镇样本

    std::string resultImagePath = "town_detection_result_optimal_hebei.png";
    cv::imwrite(resultImagePath, display_image_bgr);
    std::cout << "[Step 13] Optimal result saved to: " << resultImagePath << std::endl;

    double final_scale = std::min(1200.0 / width, 800.0 / height);
    cv::resize(display_image_bgr, display_small, cv::Size(), final_scale, final_scale);
    cv::imshow("Optimal Town Detection Result (Hebei)", display_small);

    GDALClose(poDataset);
    std::cout << "\nProcessing complete. Press any key to exit." << std::endl;
    cv::waitKey(0);

    return 0;
}