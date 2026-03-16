#include <iostream>
#include <opencv2/opencv.hpp>
#include <deque>

using namespace cv;
using namespace std;

struct LineData {
    deque<Point2f> points;
    int maxSize = 40;

    void addPoint(Point2f p) {
        if (points.size() >= maxSize) points.pop_front();
        points.push_back(p);
    }

    bool fitLineAndDraw(Mat& frame, Scalar color, float slopeLow, float slopeHigh) {
        if (points.size() < 2) return false;
        Vec4f line;
        fitLine(points, line, DIST_L2, 0, 0.01, 0.01);

        float slope = line[1] / line[0];
        slope = fabs(slope);
        if (slope < slopeLow || slope > slopeHigh) return false;

        float x0 = line[2], y0 = line[3];
        float k = y0 - slope * x0;
        y0 = frame.rows * 0.6;
        x0 = (y0 - k) / slope;
        float y1 = frame.rows;
        float x1 = (y1 - k) / slope;

        line(frame, Point(x0, y0), Point(x1, y1), color, 3, LINE_AA);
        return true;
    }
};

int main() {
    VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        cerr << "Cannot open video file!" << endl;
        return -1;
    }

    Mat frame, gray, edges;
    LineData leftLine, rightLine;
    float slopeLow = 0.4, slopeHigh = 1.5;
    int lineLength = 200;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(3, 3), 0);
        Canny(gray, edges, 40, lineLength);

        edges(Range(0, edges.rows * 0.6), Range::all()) = 0;

        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 30, 10);

        for (auto& l : lines) {
            float m = (float)(l[3] - l[1]) / (float)(l[2] - l[0]);
            if (m > slopeLow) { // 左線
                leftLine.addPoint(Point2f(l[0], l[1]));
                leftLine.addPoint(Point2f(l[2], l[3]));
            } else if (m < -slopeLow) { // 右線
                rightLine.addPoint(Point2f(l[0], l[1]));
                rightLine.addPoint(Point2f(l[2], l[3]));
            }
        }

        leftLine.fitLineAndDraw(frame, Scalar(255, 0, 0), slopeLow, slopeHigh);
        rightLine.fitLineAndDraw(frame, Scalar(0, 255, 0), slopeLow, slopeHigh);

        imshow("Lane Detection", frame);
        if ((char)waitKey(1) == 27) break; 
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
