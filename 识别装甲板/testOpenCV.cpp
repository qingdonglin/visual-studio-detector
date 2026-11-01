#include<iostream>
#include<vector>
#include<cmath>
#include<opencv2/opencv.hpp>
#include<opencv2/calib3d.hpp>   // PnP必须要这个头文件
using namespace cv;
using namespace std;


// ======================== 宏定义（只定义总尺寸！） ========================
#define ARMOR_WIDTH    0.135   // 灯条范围总宽度 13.5cm 单位米
#define ARMOR_HEIGHT   0.055   // 灯条范围总高度 5.5cm  单位米
// ========================================================================


vector<Point3f> getRealWorldPoints()
{
    vector<Point3f> objPoints;
    float half_w = ARMOR_WIDTH / 2;
    float half_h = ARMOR_HEIGHT / 2;
    // 装甲板真实物理3D点，即装甲板到底长啥样，随着移动这个值也不会变，仅仅代表装甲板的真实大小
    objPoints.push_back(Point3f(-half_w, -half_h, 0));  // 左上
    objPoints.push_back(Point3f(half_w, -half_h, 0));  // 右上
    objPoints.push_back(Point3f(half_w, half_h, 0));  // 右下
    objPoints.push_back(Point3f(-half_w, half_h, 0));  // 左下
    return objPoints;
}

// 相机内参
Mat getCameraMatrix()
{
    return (Mat_<double>(3, 3) <<
        650, 0, 320,
        0, 650, 240,
        0, 0, 1);
}

// 畸变系数
Mat getDistCoeffs()
{
    return Mat::zeros(5, 1, CV_64F);
}

// PnP解算，输入四个角点（getArmors 函数中得到的），返回距离(米)  左上 → 右上 → 右下 → 左下
double solvePnPAndGetDistance(Point2f tl, Point2f tr, Point2f br, Point2f bl)
{
    vector<Point2f> imgPoints;
    imgPoints.push_back(tl);
    imgPoints.push_back(tr);
    imgPoints.push_back(br);
    imgPoints.push_back(bl);

    Mat rvec, tvec;
    solvePnP(
        getRealWorldPoints(),// 真实世界装甲板的4个3D点（固定不变）
        imgPoints,// 画面里装甲板的4个像素点（一直在变）
        getCameraMatrix(),// 相机参数，告诉计算机：你这个摄像头长什么样、焦距多少、画面中心在哪
        getDistCoeffs(),// 畸变参数，修正画面弯曲，比如用手机广角、USB 摄像头拍照时， 画面边缘的直线会变弯，像鱼眼一样凸出来。
        rvec, tvec // 输出结果：旋转 + 位置
    );

    double x = tvec.at<double>(0);// 左右
    double y = tvec.at<double>(1);// 上下
    double z = tvec.at<double>(2);// 前后（距离）   这三个数就是pnp算出的装甲板在相机 3D 空间里的真实位置
    double distance = sqrt(x * x + y * y + z * z);

    return distance;
}





// 定义颜色阈值结构体
// 用于存储HSV颜色空间的范围参数
struct ColorRange {
    int h_min, h_max;    // 色调最小值、最大值
    int s_min, s_max;    // 饱和度最小值、最大值
    int v_min, v_max;    // 亮度最小值、最大值
};

// 定义蓝色灯条的HSV阈值
ColorRange blue_range = { 66,110,55,255,234,255 };

/*函数1.
函数功能：把图像从 BGR 转 HSV 色彩空间，根据输入图像和颜色范围，生成颜色掩码（提取指定颜色区域）
参数：img-原始图像  range-颜色阈值范围
返回值：生成的二值化掩码图像
*/
Mat getColorMask(Mat& img, ColorRange& range) {
    Mat hsv, mask;
    // 将图像从BGR色彩空间转换为HSV色彩空间
    cvtColor(img, hsv, COLOR_BGR2HSV);
    // 根据HSV范围提取目标颜色，生成二值掩码
    inRange(hsv,
        Scalar(range.h_min, range.s_min, range.v_min),
        Scalar(range.h_max, range.s_max, range.v_max),
        mask);
    return mask;
}

/*函数2.
函数功能：对掩码图像进行形态学去噪 + 面积滤波，去除小噪点干扰
参数：mask-原始二值掩码
返回值：去噪干净的二值掩码
*/
Mat denoise(Mat& mask)
{
    // 轻微竖向膨胀，只补小缺口，不野蛮放大
    Mat kernelV = getStructuringElement(MORPH_RECT, Size(2, 9));
    dilate(mask, mask, kernelV);
    // 轻度闭运算，填中间小黑点
    Mat kernelClose = getStructuringElement(MORPH_RECT, Size(2, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, kernelClose);

    // 第二步：用面积直接删掉所有小白噪点
    vector<vector<Point>> contours;//定义一个轮廓容器
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 新建一个全黑画布
    Mat cleanMask = Mat::zeros(mask.size(), CV_8UC1);

    // 只保留面积大于阈值的轮廓，小的直接扔掉
    double minArea = 20;
    for (auto& cnt : contours)
    {
        if (contourArea(cnt) > minArea)
        {
            drawContours(cleanMask, vector<vector<Point>>{cnt}, 0, Scalar(255), -1);
        }
    }

    return cleanMask;
}

/*函数3.
函数功能：查找图像中有效轮廓，提取轮廓
参数：mask-二值化掩码图像
返回值：符合面积条件的有效轮廓集合
*/
vector<vector<Point>> findLightContours(Mat& mask) {
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> valid_contours;
    for (auto& cnt : contours) {        // 正常循环
        if (contourArea(cnt) > 40) {    // 正常过滤
            valid_contours.push_back(cnt); //  正常填充
        }
    }
    return valid_contours; // 返回有数据的轮廓
}

/*函数4.    核心灯条判断
函数功能：判断旋转矩形是否符合灯条特征（宽高比、尺寸限制）
参数：rect-轮廓的最小外接旋转矩形
返回值：true-是灯条 false-不是灯条
*/
bool isLightBar(RotatedRect& rect)
{
    float w = rect.size.width;
    float h = rect.size.height;

    float tw = min(w, h);
    float th = max(w, h);
    float ratio = tw / th;

    // 宽高比限制：灯条必须细长
    if (ratio < 0.03 || ratio > 0.4)
        return false;

    // 高度范围限制
    if (th < 6 || th > 200)
        return false;

    return true;
}

/*函数5.
函数功能：从所有轮廓中筛选出符合灯条特征的旋转矩形
参数：contours-所有有效轮廓
返回值：灯条旋转矩形集合
*/
vector<RotatedRect> getLightBars(vector<vector<Point>>& contours) {
    vector<RotatedRect> light_bars;
    for (auto& cnt : contours) { // 遍历每一个轮廓

        RotatedRect rect = minAreaRect(cnt);// 生成最小外接旋转矩形

        if (isLightBar(rect)) {// 判断是否为灯条，符合则加入集合，调用子函数islightbar
            light_bars.push_back(rect);
        }
    }
    return light_bars;
}

// 灯条结构体：存储灯条完整信息，方便装甲板匹配
struct LightBar {
    RotatedRect rect;    // 灯条旋转矩形
    Point2f center;      // 灯条中心点坐标
    float height;        // 灯条高度
    float angle;         // 灯条旋转角度
};

/*函数6.
函数功能：将旋转矩形转换为灯条结构体，统一角度与数据格式      在getArmors函数中被调用
参数：rects-灯条旋转矩形集合
返回值：灯条结构体集合
*/
vector<LightBar> convertToLightBars(vector<RotatedRect>& rects) {
    vector<LightBar> bars;
    for (auto& rect : rects) {
        LightBar bar;
        bar.rect = rect;
        bar.center = rect.center;
        bar.height = max(rect.size.width, rect.size.height);
        bar.angle = rect.angle;

        // 支持倒着识别：统一把角度压到 0~90，不区分正反
        float w = rect.size.width;
        float h = rect.size.height;
        if (w > h)
            bar.angle -= 90;

        // 只保留绝对值，不判断正反
        bar.angle = abs(bar.angle);

        bars.push_back(bar);
    }
    return bars;
}

/*函数7.
函数功能：判断两个灯条是否能组成装甲板          在getArmors函数中被调用
匹配条件：角度差、距离比例、高度差、垂直偏移
参数：bar1、bar2-待匹配的两个灯条
返回值：true-可组成装甲板
*/
bool isArmor(LightBar& bar1, LightBar& bar2) {
    // 倒着、斜着都能识别：只判断角度差绝对值，不限制方向
    float angleDiff = abs(abs(bar1.angle) - abs(bar2.angle));

    // 区分左右灯条
    LightBar leftBar = (bar1.center.x < bar2.center.x) ? bar1 : bar2;
    LightBar rightBar = (bar1.center.x < bar2.center.x) ? bar2 : bar1;

    // 计算匹配所需参数
    float distance = sqrt(pow(leftBar.center.x - rightBar.center.x, 2) + pow(leftBar.center.y - rightBar.center.y, 2));//两点距离公式
    float avgHeight = (leftBar.height + rightBar.height) / 2;// (左高 + 右高) / 2，就是平均高度
    float distanceRatio = distance / avgHeight;//距离 / 平均高度
    float heightDiff = abs(leftBar.height - rightBar.height);//abs(左高 - 右高);
    float yDiff = abs(leftBar.center.y - rightBar.center.y);//abs(左y - 右y);

    bool pass = true;

    // 装甲板匹配条件
    if (angleDiff > 40) pass = false;//angleDiff > 25
    if (distanceRatio < 0.5 || distanceRatio > 8) pass = false;//距离比例不对
    if (heightDiff > avgHeight * 0.9) pass = false;//高度差太大
    if (yDiff > avgHeight * 2.0) pass = false;//垂直偏移太大

    return pass;
}

/*函数8. 辅助函数角点排序        在getLightBarEndpoints中被调用
函数功能：对旋转矩形的四个角点进行排序（上下、左右排序）
参数：pts-旋转矩形的四个角点
*/
void sortRectPoints(Point2f pts[4]) {
    sort(pts, pts + 4, [](Point2f a, Point2f b) { return a.y < b.y; });
    if (pts[0].x > pts[1].x) swap(pts[0], pts[1]);
    if (pts[2].x > pts[3].x) swap(pts[2], pts[3]);
}

/*函数9. 辅助函数
函数功能：获取灯条矩形的四个端点（左上、右上、左下、右下）
参数：rect-灯条旋转矩形
*/
void getLightBarEndpoints(RotatedRect rect, Point2f& topL, Point2f& topR, Point2f& botL, Point2f& botR) {
    Point2f pts[4];
    rect.points(pts);
    sortRectPoints(pts);//角点排序
    topL = pts[0];
    topR = pts[1];
    botL = pts[2];
    botR = pts[3];
}

/*函数10.  最终装甲板匹配与绘制
函数功能：从所有灯条中匹配出装甲板，绘制边框与交叉线，并生成外接矩形
参数：rotatedBars-灯条旋转矩形集合 img-用于绘制的图像
返回值：装甲板外接矩形集合
*/
vector<Rect> getArmors(vector<RotatedRect>& rotatedBars, Mat& img) {
    vector<LightBar> bars = convertToLightBars(rotatedBars);// 先调用 convertToLightBars
    vector<Rect> armors;
    vector<bool> used(bars.size(), false);

    for (int i = 0; i < bars.size(); ++i) {
        if (used[i]) continue;
        for (int j = i + 1; j < bars.size(); ++j) {
            if (used[j]) continue;
            if (isArmor(bars[i], bars[j])) {//调用isarmor函数

                // 区分左右灯条
                LightBar leftBar = (bars[i].center.x < bars[j].center.x) ? bars[i] : bars[j];
                LightBar rightBar = (bars[i].center.x < bars[j].center.x) ? bars[j] : bars[i];

                // 获取两个灯条的四个角点
                Point2f lt1, rt1, lb1, rb1;
                Point2f lt2, rt2, lb2, rb2;
                getLightBarEndpoints(leftBar.rect, lt1, rt1, lb1, rb1);//取灯条四个角的点
                getLightBarEndpoints(rightBar.rect, lt2, rt2, lb2, rb2);

                // 组合装甲板四个角点
                Point2f tl = lt1;
                Point2f tr = rt2;
                Point2f bl = lb1;
                Point2f br = rb2;

                // 绘制装甲板外框
                line(img, tl, tr, Scalar(0, 255, 0), 2);
                line(img, tr, br, Scalar(0, 255, 0), 2);
                line(img, br, bl, Scalar(0, 255, 0), 2);
                line(img, bl, tl, Scalar(0, 255, 0), 2);

                // 绘制交叉对角线
                line(img, tl, br, Scalar(0, 255, 0), 1);
                line(img, tr, bl, Scalar(0, 255, 0), 1);

                // 绘制装甲板中心点
                circle(img, (tl + br) * 0.5f, 4, Scalar(0, 255, 255), -1);
                // PnP计算距离并显示
                double dis = solvePnPAndGetDistance(tl, tr, br, bl);
                putText(img, to_string(dis).substr(0, 4) + "m", Point(tl.x, tl.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
                // 生成装甲板外接正矩形
                int minX = min({ (int)tl.x, (int)tr.x, (int)bl.x, (int)br.x });
                int maxX = max({ (int)tl.x, (int)tr.x, (int)bl.x, (int)br.x });
                int minY = min({ (int)tl.y, (int)tr.y, (int)bl.y, (int)br.y });
                int maxY = max({ (int)tl.y, (int)tr.y, (int)bl.y, (int)br.y });
                armors.push_back(Rect(minX, minY, maxX - minX, maxY - minY));

                // 标记已使用，避免重复匹配
                used[i] = used[j] = true;
                break;
            }
        }
    }
    return armors;
}

// 主函数：程序入口
int main()
{
    // 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "摄像头打开失败" << endl;
        return -1;
    }
    // 摄像头参数配置：曝光、白平衡等
    //cap.set(CAP_PROP_AUTO_EXPOSURE, 0);// 关闭自动曝光
    cap.set(CAP_PROP_EXPOSURE, -5); // 手动曝光（-6更暗，灯条更亮）
    //cap.set(CAP_PROP_AUTO_WB, 0);// 关闭自动白平衡
    //cap.set(CAP_PROP_WB_TEMPERATURE, 4500); // 手动白平衡*/

    // 循环读取每一帧
    while (1)
    {
        Mat img;
        cap >> img;

        // 1. 生成颜色掩码
        Mat mask = getColorMask(img, blue_range);
        imshow("1,颜色掩码", mask);

        // 2. 掩码去噪
        mask = denoise(mask);
        imshow("2.去噪后掩码", mask);

        // 3. 查找有效轮廓
        vector<vector<Point>> contours = findLightContours(mask);
        Mat contour_img = img.clone();
        drawContours(contour_img, contours, -1, Scalar(0, 0, 255), 2);//用红色线画出轮廓
        imshow("3.轮廓", contour_img);

        // 4. 筛选灯条并绘制
        vector<RotatedRect> light_bars = getLightBars(contours);
        Mat light_img = img.clone();
        for (auto& bar : light_bars) {
            Point2f pts[4];
            bar.points(pts);
            for (int i = 0; i < 4; i++) {
                line(light_img, pts[i], pts[(i + 1) % 4], Scalar(0, 255, 0), 2);
            }
        }
        imshow("4.发光条", light_img);

        // 打印灯条信息（可选）
        for (int i = 0; i < light_bars.size(); i++) {
            float w = light_bars[i].size.width;
            float h = light_bars[i].size.height;
            float tw = min(w, h);
            float th = max(w, h);
            cout << "灯条" << i << ": 中心(" << light_bars[i].center.x << "," << light_bars[i].center.y
                << ") 宽=" << w << " 高=" << h
                << " angle=" << light_bars[i].angle
                << " 宽高比=" << tw / th << endl;
        }
        cout << "=========================" << endl;

        // 5. 匹配装甲板并绘制结果
        Mat result = img.clone(); // 先克隆
        vector<Rect> armors = getArmors(light_bars, result); // 画线画在result上

        for (auto& armor : armors) {
            rectangle(result, armor, Scalar(255, 0, 0), 2);
        }
        imshow("5. 最终结果", result);

        // ESC 退出
        if (waitKey(1) == 27)
            break;
    }

    // 释放资源
    cap.release();
    destroyAllWindows();
    return 0;
}