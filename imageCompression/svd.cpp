#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

int main() {
    cv::Mat img = cv::imread("nature.jpg",1);
    int k = 345;
    float R = (float)k*(1+(float)img.rows+(float)img.cols)/((float)img.rows*(float)img.cols);
    cv::Mat uu = cv::Mat::zeros(img.rows,k,CV_32FC1);
    cv::Mat ss = cv::Mat::zeros(k,k,CV_32FC1);
    cv::Mat vvt = cv::Mat::zeros(k,img.cols,CV_32FC1);

    cv::Mat blue = cv::Mat::zeros(img.rows,img.cols,CV_32FC1);
    cv::Mat green = cv::Mat::zeros(img.rows,img.cols,CV_32FC1);
    cv::Mat red = cv::Mat::zeros(img.rows,img.cols,CV_32FC1);
    if (img.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
    }

    cv::Mat imgBlue, imgRed, imgGreen;
    std::vector<cv::Mat> channels;
    cv::split(img, channels);//split BGR
	imgBlue = channels.at(0);
	imgGreen = channels.at(1);
	imgRed = channels.at(2);
    imgBlue.convertTo(imgBlue,CV_32FC1);
    imgGreen.convertTo(imgGreen,CV_32FC1);
    imgRed.convertTo(imgRed,CV_32FC1);
    // the image store in usinged char, but SVD input is float or double.
    cv::Mat sBlue,uBlue,vtBlue;
    imgBlue.convertTo(blue, CV_32FC1);
    //std::cout<<"do SVD"<<std::endl;
    //SVD needs much more time, this print just let you know it's computing......
    cv::SVD::compute(imgBlue,sBlue,uBlue,vtBlue,4);
    //std::cout<<uBlue.size<<std::endl;
    //SVD needs much more time, this print just let you know it's computing......
    for (int i = 0;i < imgBlue.rows;i++)
    {
        if(i < k) ss.at<float>(i,i) = sBlue.at<float>(i);
        for (int j = 0;j < imgBlue.cols;j++)
        {
            if(j < k) uu.at<float>(i,j) = uBlue.at<float>(i,j);
            if(i < k) vvt.at<float>(i,j) = vtBlue.at<float>(i,j);
        }
    }
    blue = uu*ss*vvt;


    cv::Mat uGreen,sGreen,vtGreen;
    imgGreen.convertTo(green, CV_32FC1);
    //std::cout<<"do SVD"<<std::endl;
    //SVD needs much more time, this print just let you know it's computing......
    cv::SVD::compute(imgGreen,sGreen,uGreen,vtGreen,4);
    //std::cout<<uGreen.size<<std::endl;
    //SVD needs much more time, this print just let you know it's computing......
    for (int i = 0;i < imgGreen.rows;i++)
    {
        if(i < k) ss.at<float>(i,i) = sGreen.at<float>(i);
        for (int j = 0;j < imgGreen.cols;j++)
        {
            if(j < k) uu.at<float>(i,j) = uGreen.at<float>(i,j);
            if(i < k) vvt.at<float>(i,j) = vtGreen.at<float>(i,j);
        }
    }
    green = uu*ss*vvt;

    cv::SVD::compute(imgGreen,sGreen,uGreen,vtGreen,4);
    cv::Mat uRed,sRed,vtRed;
    //std::cout<<"do SVD"<<std::endl;
    //SVD needs much more time, this print just let you know it's computing......
    cv::SVD::compute(imgRed,sRed,uRed,vtRed,4);
    //std::cout<<uRed.size<<std::endl;
    //SVD needs much more time, this print just let you know it's computing......
    for (int i = 0;i < imgGreen.rows;i++)
    {
        if(i < k) ss.at<float>(i,i) = sRed.at<float>(i);
        for (int j = 0;j < imgGreen.cols;j++)
        {
            if(j < k) uu.at<float>(i,j) = uRed.at<float>(i,j);
            if(i < k) vvt.at<float>(i,j) = vtRed.at<float>(i,j);
        }
    }
    red = uu*ss*vvt;
    blue.convertTo(blue, CV_8U);
    green.convertTo(green, CV_8U);
    red.convertTo(red, CV_8U);

    channels.at(0) = blue;
	channels.at(1) = green;
	channels.at(2) = red;
	cv::Mat temp;
	merge(channels, temp);
    temp.convertTo(temp, CV_8U);
    std::cout<< "Ratio is "<< R <<std::endl;
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow("Display window", temp);
    //cv::imwrite("gray_k.jpg",temp);
    cv::waitKey(0);
	return 0;
}
