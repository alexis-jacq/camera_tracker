#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera

    if(!cap.isOpened())  // check if we succeeded
    {
        return -1;
    }

    Mat gframe; // gray frame that will be used for goodfeaturtotrack

    namedWindow("edges",1);

    while(true)
    {
        Mat frame;

        cap >> frame; // get a new frame from camera

        cvtColor(frame, gframe, CV_BGR2GRAY); // color -> gray

        // transformation
        //GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        //Canny(edges, edges, 0, 30, 3);

        imshow("edges", gframe);

        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor

    return 0;
}
