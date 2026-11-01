#include<iostream>
#include<vector>
#include<cmath>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

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
















struct ColorRange {
    int h_min, h_max;
    int s_min, s_max;
    int v_min, v_max;
};


//函数1颜色分割
Mat getColorMask(Mat& img, ColorRange& range) {

    Mat hsv, mask;//颜色分割
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv,
        Scalar(range.h_min, range.s_min, range.v_min),
        Scalar(range.h_max, range.s_max, range.v_max),
        mask);
    return mask;
    imshow("1,颜色掩码", mask);
}
//函数2去噪
Mat denoise(Mat& mask) {//去噪
    Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 1));
    erode(mask, mask, kernel);
    dilate(mask, mask, kernel);
    return mask;
    imshow("2.去噪后掩码", mask);
}
//函数3找轮廓
vector<vector<Point>>findLightContours(Mat& mask) {//找轮廓 —— 识别发光条的形状
    vector<vector<Point>>contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>>valid_contours;
    for (auto& cnt : contours) {
        if (contourArea(cnt) > 25) {
            valid_contours.push_back(cnt);



        }



    }
    return valid_contours;//主要是筛选发光条 —— 排除不是发光条的轮廓
    Mat contour_img = img.clone();
    drawContours(contour_img, contours, -1, Scalar(0, 0, 255), 2);
    imshow("3.轮廓", contour_img);
}

//函数4：筛选发光条
bool isLightBar(RotatedRect& rect) {
    // 取宽和高中的较小值为“实际宽”，较大值为“实际高”
    float real_width = min(rect.size.width, rect.size.height);
    float real_height = max(rect.size.width, rect.size.height);
    float ratio = real_width / real_height;  // 确保 ratio ≤ 1

    // 调整条件（根据实际打印的参数设置）
    return (ratio < 0.25) && (real_height > 57);  // 用实际高度判断
}



vector<RotatedRect>getLightBars(vector<vector<Point>>& contours) {//第二步： 从轮廓中筛选出发光条的外接矩形
    vector<RotatedRect>light_bars;
    for (auto& cnt : contours) {
        RotatedRect rect = minAreaRect(cnt);
        if (isLightBar(rect)) {
            light_bars.push_back(rect);
        }




    }

    return light_bars;
}


// 5. 匹配装甲板（两个平行发光条）

// 存储发光条的额外信息（方便排序）
struct LightBar {
    RotatedRect rect;  // 旋转矩形
    Point2f center;    // 中心坐标
    float height;      // 高度（长边）
    float angle;       // 角度
};

// 函数5.1：转换RotatedRect为LightBar（统一宽高定义：height为长边）
vector<LightBar> convertToLightBars(vector<RotatedRect>& rects) {
    vector<LightBar> bars;
    for (auto& rect : rects) {
        LightBar bar;
        bar.rect = rect;
        bar.center = rect.center;
        // 统一高度为长边（避免角度导致的宽高颠倒）
        bar.height = max(rect.size.width, rect.size.height);
        bar.angle = rect.angle;
        bars.push_back(bar);
    }
    return bars;
}

// 函数5.2：优化后的装甲板匹配条件（针对多个并排装甲板）
bool isArmor(LightBar& bar1, LightBar& bar2) {
    // 1. 角度差：近似平行（<10°）
    float angleDiff = abs(bar1.angle - bar2.angle);
    if (angleDiff > 10) return false;

    // 2. 左右位置：必须一个在左，一个在右（避免上下组合）
    if (bar1.center.x == bar2.center.x) return false;  // 排除垂直对齐
    LightBar leftBar = (bar1.center.x < bar2.center.x) ? bar1 : bar2;
    LightBar rightBar = (bar1.center.x < bar2.center.x) ? bar2 : bar1;

    // 3. 中心距与高度比：1.2~2倍（根据实际装甲板尺寸调整）
    float distance = sqrt(pow(leftBar.center.x - rightBar.center.x, 2) +
        pow(leftBar.center.y - rightBar.center.y, 2));
    float avgHeight = (leftBar.height + rightBar.height) / 2;
    float distanceRatio = distance / avgHeight;
    if (distanceRatio < 2 || distanceRatio > 5) return false;

    // 4. 高度差：<30%（大小相似）
    if (abs(leftBar.height - rightBar.height) > avgHeight * 0.3) return false;

    // 5. 上下对齐：中心y坐标差<高度的1/3（避免左右条上下偏移过大）
    float yDiff = abs(leftBar.center.y - rightBar.center.y);
    if (yDiff > avgHeight / 3) return false;

    return true;
}

// 函数5.3：从发光条中筛选装甲板（去重）
vector<Rect> getArmors(vector<RotatedRect>& rotatedBars) {
    vector<LightBar> bars = convertToLightBars(rotatedBars);
    vector<Rect> armors;
    vector<bool> used(bars.size(), false);  // 标记已使用的发光条

    for (int i = 0; i < bars.size(); ++i) {
        if (used[i]) continue;  // 跳过已匹配的发光条
        for (int j = i + 1; j < bars.size(); ++j) {
            if (used[j]) continue;
            if (isArmor(bars[i], bars[j])) {
                // 计算装甲板外接矩形
                Point2f p1[4], p2[4];
                bars[i].rect.points(p1);
                bars[j].rect.points(p2);
                int minX = 1e5, maxX = 0, minY = 1e5, maxY = 0;
                for (int k = 0; k < 4; ++k) {
                    minX = min(minX, (int)p1[k].x);
                    maxX = max(maxX, (int)p1[k].x);
                    minY = min(minY, (int)p1[k].y);
                    maxY = max(maxY, (int)p1[k].y);
                    minX = min(minX, (int)p2[k].x);
                    maxX = max(maxX, (int)p2[k].x);
                    minY = min(minY, (int)p2[k].y);
                    maxY = max(maxY, (int)p2[k].y);
                }
                armors.push_back(Rect(minX, minY, maxX - minX, maxY - minY));
                used[i] = true;  // 标记为已使用，避免重复匹配
                used[j] = true;
                break;  // 一个发光条只匹配一次
            }
        }
    }
    return armors;
}































int main()
{












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

    //读取图像
    Mat img = imread("zj.jpg");





    //蓝色发光条的hsv范围
    ColorRange blue_range = { 0,111,0,255,220,255 };
   //步骤1：提取蓝色掩码
    Mat mask = getColorMask(img, blue_range);//函数1

    
    //步骤2：去噪
    mask = denoise(mask);//函数2
 
    
    //步骤3：找轮廓
    vector<vector<Point>> contours = findLightContours(mask);//函数3
    
    
    //步骤4：筛选发光条
    vector<RotatedRect>light_bars = getLightBars(contours);
    Mat light_img = img.clone();
    for (auto& bar : light_bars) {
        Point2f pts[4];
        bar.points(pts);

        for (int i = 0; i < 4; i++) {
            line(light_img, pts[i], pts[(i + 1) % 4], Scalar(0, 255, 0), 2);
        }
    }
    imshow("4.发光条", light_img);
    // 步骤5：找装甲板
    vector<Rect> armors = getArmors(light_bars);
    Mat result = img.clone();
    for (auto& armor : armors) {
        rectangle(result, armor, Scalar(0, 255, 0), 2);  // 蓝色画装甲板
    }
    imshow("5. 最终结果", result);









    waitKey(0);
    return 0;
}
