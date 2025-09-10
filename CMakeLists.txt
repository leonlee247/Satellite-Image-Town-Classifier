# CMake 最低版本要求
cmake_minimum_required(VERSION 3.15)

# 定义项目名称和 C++ 标准
project(SceneClassifier LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 寻找我们需要的“外部军团” (GDAL 和 OpenCV) ---
# 保留vcpkg的工具链，以便它能为我们找到GDAL
find_package(GDAL REQUIRED)

# 直接指定官方OpenCV的路径，绕开vcpkg
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV 4.12.0 REQUIRED COMPONENTS core imgproc highgui ml)

# --- 最终的“立法”部分：定义两个独立的王国 ---

# === 王国一：场景分类器 (SceneClassifier) ===
# 这条法律定义了我们的主程序，它的唯一源文件是 main.cpp
add_executable(SceneClassifier main.cpp)

# 为 SceneClassifier 王国授权，链接它需要的库
target_include_directories(SceneClassifier PRIVATE ${GDAL_INCLUDE_DIRS})
target_link_libraries(SceneClassifier PRIVATE
    ${OpenCV_LIBS}
    GDAL::GDAL
)
# 为 Windows 平台设置
if(WIN32)
    target_compile_definitions(SceneClassifier PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()


# === 王国二：样本选取器 (SampleSelector) ===
# 这条法律定义了我们的工具程序，它的唯一源文件是 select_samples.cpp
add_executable(SampleSelector select_samples.cpp)

# 同样为 SampleSelector 王国授权，链接它需要的库
target_include_directories(SampleSelector PRIVATE ${GDAL_INCLUDE_DIRS})
target_link_libraries(SampleSelector PRIVATE
    ${OpenCV_LIBS}
    GDAL::GDAL
)


# --- 立法报告 ---
message(STATUS "--- CMake Configuration Report ---")
message(STATUS "Successfully defined 2 executables:")
message(STATUS "  1. SceneClassifier (from main.cpp)")
message(STATUS "  2. SampleSelector (from select_samples.cpp)")
message(STATUS "GDAL found: ${GDAL_VERSION}")
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "------------------------------------")