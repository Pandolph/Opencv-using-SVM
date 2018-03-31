//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include "opencv2/imgcodecs.hpp"
//#include <opencv2/highgui.hpp>
//#include <opencv2/ml.hpp>
#include <iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
    Mat srcImg = imread("/Users/pan/Desktop/SVM/SVM/test.tif");
    Mat desImg = srcImg.clone();
    imshow("rawImg", srcImg);
    // 选取目标区域和背景区域
    Mat BackImg = srcImg(Rect(199, 0, 30, 30)); //2D, Rect(x,y,width,height)
    Mat ForeImg = srcImg(Rect(38, 95, 30, 30));
    int k = srcImg.channels();
    printf("k=%d",k);
    
    // 初始化训练数据
    Mat trainingDataMat = ForeImg.clone().reshape(1, ForeImg.cols*ForeImg.rows); //reshape(channels, rows) which means it will be rows = 900; cols = 3;
    print(trainingDataMat);
    //在这里直接存入背景像素点，或者像下边一个一个点存入也可以
    trainingDataMat.push_back(BackImg.clone().reshape(1,BackImg.cols*BackImg.rows));
    trainingDataMat.convertTo(trainingDataMat, CV_32FC1);
    
    // 初始化标签，分别给两种标签辅助，虽然这里memset已经全部初始化为1了，可是这里的1是浮点数
    int *labels = new int[ForeImg.cols*ForeImg.rows + BackImg.cols*BackImg.rows];
    memset(labels, 1, sizeof(int)*(ForeImg.cols*ForeImg.rows + BackImg.cols*BackImg.rows));
    for (int i = 0; i < ForeImg.rows; ++i)
        for (int j = 0; j < ForeImg.cols; ++j){
            labels[i*ForeImg.cols + j] = 1;
        }
    for (int h = 0; h<BackImg.rows; h++)
    {
        for (int w = 0; w<BackImg.cols; w++)
        {
            labels[ForeImg.cols*ForeImg.rows + h*BackImg.cols + w] = -1;
        }
    }
    Mat labelsMat = Mat(ForeImg.cols*ForeImg.rows + BackImg.cols*BackImg.rows, 1, CV_32SC1, labels);
    
    // 可以将数据写入文件来检查是否正确
    FileStorage fs("/Users/pan/Desktop/SVM/SVM/data.xml", FileStorage::WRITE);
    fs << "traindata" << trainingDataMat << "labels" << labelsMat;
    
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e5, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    
    // 开始分类
    Vec3b black(0, 0, 0), white(255, 255, 255);
    for (int i = 0; i < desImg.rows; ++i)
    {
        uchar* p_sample = desImg.ptr<uchar>(i); // p_sample是desImg第i行的头指针
        for (int j = 0; j < desImg.cols; ++j)
        {
            Mat sampleMat(1, 3, CV_32FC1);
            sampleMat.at<float>(0, 0) = p_sample[3 * j + 0];
            sampleMat.at<float>(0, 1) = p_sample[3 * j + 1];
            sampleMat.at<float>(0, 2) = p_sample[3 * j + 2];
            
            float response = svm->predict(sampleMat);
            
            if (response == 1)
                desImg.at<Vec3b>(i, j) = white;
            else if (response == -1)
                desImg.at<Vec3b>(i, j) = black;
        }
    }
    
    imwrite("result.jpg", desImg);
    imshow("resImg", desImg);
    
    waitKey(0);
    return 0;
}
