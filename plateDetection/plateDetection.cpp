
#include <iostream>
#include <cmath>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

bool verifySizes(cv::RotatedRect mr) {
    float error = 0.9;
    //Spain car plate size: 52x11 aspect 4,7272
    //China car plate size: 440mm*140mm¡Aaspect 3.142857
    float aspect = 4;
    //Set a min and max area. All other patchs are discarded
    //int min= 1*aspect*1; // minimum area
    //int max= 2000*aspect*2000; // maximum area
    int min= 44*14*1; // minimum area
    int max= 44*14*10; // maximum area
    //Get only patchs that match to a respect ratio.
    float rmin= aspect-aspect*error;
    float rmax= aspect+aspect*error;

    int area= mr.size.height * mr.size.width;
    float r = (float)mr.size.width / (float)mr.size.height;
    if(r < 1) r = (float)mr.size.height / (float)mr.size.width;
    if(( area < min || area > max ) || ( r < rmin || r > rmax )) return false;
    else return true;
}

cv::Mat showResultMat(cv::Mat src, cv::Size rect_size, cv::Point2f center, int index) {
    cv::Mat img_crop;
    getRectSubPix(src, rect_size, center, img_crop);

    cv::Mat resultResized;
    resultResized.create(36, 136, CV_8UC3);
    cv::resize(img_crop, resultResized, resultResized.size(), 0, 0, cv::INTER_CUBIC);

    return resultResized;
}

int main() {
    cv::Mat img, gray, dst, roi, sobelX, sobelY, result, resultMat, img_rotated, rotmat;
    std::vector<cv::Vec4i> hierarchy;
    std::vector< std::vector<cv::Point>> contours;
    img = cv::imread("test.jpg",1);
    cv::GaussianBlur(img, img, cv::Size( 5, 5 ), 0, 0 );
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Sobel(gray, sobelX, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs( sobelX, sobelX );
    cv::Sobel(gray, sobelY, CV_8U, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs( sobelY, sobelY );
    addWeighted( sobelX, 0.5, sobelY, 0.5, 0, dst );
    cv::threshold(dst, dst, 100, 255, cv::THRESH_BINARY| cv::THRESH_OTSU);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(23, 3), cv::Point(-1, -1));
    cv::morphologyEx(dst, dst, 3, element); //3 = MOP_CLOSE
	cv::findContours(dst, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
    std::vector<cv::RotatedRect> rects;
    //Remove patch that are no inside limits of aspect ratio and area.
    int t = 0;
    while (itc != contours.end()) {
        //Create bounding rect of object
        cv::RotatedRect mr = cv::minAreaRect(cv::Mat(*itc));

        //large the rect for more
        if(!verifySizes(mr)) itc = contours.erase(itc);
        else {
            ++itc;
            rects.push_back(mr);
        }
    }

    int k = 1, m_debug = 0, m_angle= 20;
    std::vector<cv::Mat> resultVec;
    for(int i=0; i< rects.size(); i++) {
        cv::RotatedRect minRect = rects[i];
        if(verifySizes(minRect)) {
            if(m_debug) {
                cv::Point2f rect_points[4];
                minRect.points( rect_points );
                for( int j = 0; j < 4; j++ )
                    line( result, rect_points[j], rect_points[(j+1)%4], cv::Scalar(0,255,255), 1, 8 );
            }

            float r = (float)minRect.size.width / (float)minRect.size.height;
            float angle = minRect.angle;
            cv::Size rect_size = minRect.size;
            if (r < 1) {
                angle = 90 + angle;
                std::swap(rect_size.width, rect_size.height);
            }
            if (angle - m_angle < 0 && angle + m_angle > 0) {
                //Create and rotate image
                rotmat = getRotationMatrix2D(minRect.center, angle, 1);
                cv::warpAffine(img, img_rotated, rotmat, img.size(), cv::INTER_CUBIC);

                resultMat = showResultMat(img_rotated, rect_size, minRect.center, k++);

                resultVec.push_back(resultMat);
            }
        }
    }

    cv::drawContours(img, contours, -1, cv::Scalar(0,255,255), 1);
    cv::imshow("Sobel", resultVec.at(0));

    cv::waitKey(0);
    return 0;
}
