#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "opencv2/ocl/ocl.hpp"
#include <time.h>

using namespace cv;
using namespace std;

int main() {
    timespec tstart, tend;

    Size win_size(128, 64);
//    Size win_stride(win_stride_width, win_stride_height);
    ocl::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9);

    std::vector<float> svm_vectors;
    std::ifstream file("/home/igor/Documents/autorally-detection/svm.txt");
    std::string line;
    while ( std::getline(file, line) ) {
        if ( !line.empty() )
            svm_vectors.push_back(stof(line));
    }
    gpu_hog.setSVMDetector(svm_vectors);

    VideoCapture cap("/home/igor/py-faster-rcnn/data/demo/autorallyDetection2.mp4");
    if ( !cap.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the video file" << endl;
        return -1;
    }
    namedWindow("video");
    Mat frame, frame_aux;
    ocl::oclMat gpu_img;
    vector<Rect> found;

    while(true){
        cap >> frame;
        cvtColor(frame, frame_aux, CV_BGR2BGRA);

        gpu_img.upload(frame_aux);
        clock_gettime(CLOCK_REALTIME, &tstart);
        gpu_hog.detectMultiScale(gpu_img, found, 0, Size(16,16), Size(0,0), 1.05);

        for (size_t i = 0; i < found.size(); i++)
        {
            Rect r = found[i];
            rectangle(frame, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
        }
        clock_gettime(CLOCK_REALTIME, &tend);
        double dif = tend.tv_nsec - tstart.tv_nsec;
        cout << dif*1e-6 << " ms" << endl;

        imshow("video", frame);
        if(waitKey(1) == 27)
            break;
    }
    return 0;
}