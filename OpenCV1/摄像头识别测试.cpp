#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 打开默认摄像头（参数0对应内置摄像头，部分设备可能需要1、2等）
    VideoCapture cap(0);

    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头，请检查设备或权限！" << endl;
        return -1;
    }

    Mat frame; // 存储摄像头帧画面
    while (true) {
        // 读取一帧画面
        cap.read(frame);

        // 检查画面是否读取成功
        if (frame.empty()) {
            cerr << "无法获取画面，退出程序！" << endl;
            break;
        }

        // 显示画面（窗口名"Camera"）
        imshow("Camera", frame);

        // 按下ESC键退出（等待30毫秒检测按键）
        char c = (char)waitKey(30);
        if (c == 27) { // 27是ESC的ASCII码
            break;
        }
    }

    // 释放资源
    cap.release();
    destroyAllWindows();
    return 0;
}