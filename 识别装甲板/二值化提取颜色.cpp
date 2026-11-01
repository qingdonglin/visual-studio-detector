#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
using namespace cv;

// ===== 参数 =====
int alpha = 79;      // 亮度（模拟曝光）
int thresh_bin = 113;      // 灰度二值化阈值
int morph_size = 0;        // 闭运算核（连点阵散点）
int color_delta = 17;       // 红蓝色差
int min_area = 340;       // 面积滤波：最小发光区域面积
int final_close = 7;        // ★ 最后一步的闭运算核（焊死红/蓝掩码里的空隙）

Mat src;

// ================= 去噪：面积滤波（只保留够大的块）=================
Mat removeSmallBlobs(const Mat& mask, int min_area)
{
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat clean = Mat::zeros(mask.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contourArea(contours[i]) >= min_area) {
            drawContours(clean, contours, (int)i, Scalar(255), FILLED);
        }
    }
    return clean;
}

void onTrackbar(int, void*)
{
    if (src.empty()) return;

    // ① 调亮度
    Mat work;
    src.convertTo(work, -1, alpha / 100.0, 0);

    // ② 灰度二值化
    Mat gray, binary;
    cvtColor(work, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, thresh_bin, 255, THRESH_BINARY);

    // ③ 闭运算（点阵散点 → 整块）
    Mat closed = binary.clone();
    if (morph_size > 1) {
        Mat k = getStructuringElement(MORPH_RECT, Size(morph_size, morph_size));
        morphologyEx(binary, closed, MORPH_CLOSE, k);
    }

    // ④ 二值化后就去噪：面积滤波 → 干净的发光区域
    Mat light_region = removeSmallBlobs(closed, min_area);

    // ⑤ 红蓝掩码（差值法）
    std::vector<Mat> bgr;
    split(work, bgr);
    Mat b32, r32;
    bgr[0].convertTo(b32, CV_32S);
    bgr[2].convertTo(r32, CV_32S);
    Mat red_mask, blue_mask;
    compare(r32 - b32, color_delta, red_mask, CMP_GT);
    compare(b32 - r32, color_delta, blue_mask, CMP_GT);

    // ⑥ 颜色 & 干净的发光区域
    Mat red = red_mask & light_region;
    Mat blue = blue_mask & light_region;

    // ⑦ ★★ 最后再闭一次：把红/蓝掩码里的空隙、断点焊上 ★★
    Mat red_c = red.clone(), blue_c = blue.clone();
    if (final_close > 1) {
        Mat k2 = getStructuringElement(MORPH_ELLIPSE, Size(final_close, final_close));
        morphologyEx(red, red_c, MORPH_CLOSE, k2);
        morphologyEx(blue, blue_c, MORPH_CLOSE, k2);
    }

    // ⑧ 叠加（用补洞后的掩码上色）
    Mat overlay = work.clone();
    overlay.setTo(Scalar(0, 0, 255), red_c);
    overlay.setTo(Scalar(255, 0, 0), blue_c);

    // ⑨ 显示
    imshow("0-原图(alpha)", work);
    imshow("1-二值化", binary);
    imshow("2-闭运算", closed);
    imshow("3-去噪后发光区", light_region);
    imshow("4-红掩码(闭前)", red);
    imshow("5-红掩码(闭后)", red_c);      // ★ 对比看空隙
    imshow("6-蓝掩码(闭后)", blue_c);
    imshow("7-叠加", overlay);

    printf("\ralpha=%3d 阈值=%3d 闭核=%2d 色差=%3d 最小面积=%4d 末次闭核=%2d | 发光区=%6d 红=%6d→%6d 蓝=%6d→%6d      ",
        alpha, thresh_bin, morph_size, color_delta, min_area, final_close,
        countNonZero(light_region),
        countNonZero(red), countNonZero(red_c),
        countNonZero(blue), countNonZero(blue_c));
    fflush(stdout);
}

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "b.jpg";
    src = imread(path);
    if (src.empty()) { printf("读图失败：%s\n", path); return -1; }

    namedWindow("滑动条", WINDOW_AUTOSIZE);
    createTrackbar("alpha(100=原图)", "滑动条", &alpha, 200, onTrackbar);
    createTrackbar("二值化阈值", "滑动条", &thresh_bin, 255, onTrackbar);
    createTrackbar("闭运算核", "滑动条", &morph_size, 31, onTrackbar);
    createTrackbar("红蓝色差", "滑动条", &color_delta, 150, onTrackbar);
    createTrackbar("最小发光面积", "滑动条", &min_area, 2000, onTrackbar);
    createTrackbar("末次闭运算核", "滑动条", &final_close, 31, onTrackbar);   // ★ 新增

    onTrackbar(0, 0);
    printf("\n按 ESC 退出\n");
    while (waitKey(0) != 27) {}
    return 0;
}