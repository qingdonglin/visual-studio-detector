#include <opencv2/opencv.hpp>
using namespace cv;

// 全局变量：HSV阈值（初始值可先填一个大概范围）
int h_min = 90, h_max = 120;  // 蓝色H值参考范围
int s_min = 100, s_max = 255;
int v_min = 100, v_max = 255;

// 原图和HSV图（全局变量，方便回调函数访问）
Mat src_img, hsv_img;

// 滑动条回调函数（值变化时更新掩码）
void onTrackbar(int, void*) {
    // 根据当前滑动条值生成掩码
    Mat mask;
    inRange(hsv_img,
        Scalar(h_min, s_min, v_min),
        Scalar(h_max, s_max, v_max),
        mask);

    // 在"掩码效果"窗口显示结果
    imshow("掩码效果", mask);
}

int main() {
   
    Mat src_img = imread("a.jpg");


    // 1. 读取原图（替换为你的图片路径）
  
    if (src_img.empty()) {
        printf("无法读取图片！请检查路径是否正确。");
        return -1;
    }

    // 2. 转换为HSV格式（用于颜色分割）
    cvtColor(src_img, hsv_img, COLOR_BGR2HSV);

    // 3. 创建两个窗口：一个放滑动条，一个放掩码效果
    namedWindow("滑动条调节", WINDOW_AUTOSIZE);  // 滑动条窗口
    namedWindow("掩码效果", WINDOW_AUTOSIZE);    // 掩码显示窗口

    // 4. 在"滑动条调节"窗口创建6个滑动条（H/S/V的min和max）
    // 参数：滑动条名称、绑定的窗口、变量地址、最大值、回调函数
    createTrackbar("H_min", "滑动条调节", &h_min, 180, onTrackbar);
    createTrackbar("H_max", "滑动条调节", &h_max, 180, onTrackbar);
    createTrackbar("S_min", "滑动条调节", &s_min, 255, onTrackbar);
    createTrackbar("S_max", "滑动条调节", &s_max, 255, onTrackbar);
    createTrackbar("V_min", "滑动条调节", &v_min, 255, onTrackbar);
    createTrackbar("V_max", "滑动条调节", &v_max, 255, onTrackbar);

    // 5. 初始化显示一次掩码（否则"掩码效果"窗口会空白）
    onTrackbar(0, 0);

    // 6. 显示原图（可选，方便对比）
    imshow("原图", src_img);

    // 等待按键（ESC退出）
    waitKey(0);

    // 关闭所有窗口
    destroyAllWindows();
    return 0;
}