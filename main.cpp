#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>
#include "opencv2/ocl/ocl.hpp"
#include <time.h>

using namespace cv;
using namespace std;

vector<Rect> nms();

int main() {
    timespec tstart, tend;
    double scale;
    double resize_scale;
    int win_width = 128;
    int win_stride_width, win_stride_height;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;
    win_stride_width = 16;
    win_stride_height = 16;
    gr_threshold = 8;
    nlevels = 13;
    hit_threshold = 0.7;
    scale = 1.05;
    gamma_corr = false;


    Size win_size(128, 64);
    Size win_stride(win_stride_width, win_stride_height);
//    Size win_stride(win_stride_width, win_stride_height);
//    ocl::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9);
    ocl::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9);

    std::vector<float> svm_vectors;
    std::ifstream file("/home/igor/Documents/autorally-detection/svm.txt");
    std::string line;
    while ( std::getline(file, line) ) {
        if ( !line.empty() )
            svm_vectors.push_back(stof(line));
    }
    gpu_hog.setSVMDetector(svm_vectors);

    VideoCapture cap("/home/igor/Documents/autorally-detection/autorally_database/Videos/0002.mp4");
    if ( !cap.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the video file" << endl;
        return -1;
    }
    namedWindow("video");
    Mat frame, frame_aux;
    ocl::oclMat gpu_img;
    vector<Rect> found;
    resize_scale = 3;
    while(true){
        cap >> frame;
        Size sz((int)((double)frame.cols/resize_scale), (int)((double)frame.rows/resize_scale));
        resize(frame, frame, sz);

//        cvtColor(frame, frame_aux, CV_BGR2BGRA);
        cvtColor(frame, frame_aux, CV_BGR2GRAY);
//        gpu_hog.nlevels = nlevels;
        gpu_img.upload(frame_aux);
        clock_gettime(CLOCK_REALTIME, &tstart);
//        gpu_hog.detectMultiScale(gpu_img, found, 0, Size(16,16), Size(0,0), 1.05);
        gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride,
                                 Size(0, 0), scale, gr_threshold);
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