#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <facetracker.hpp>

using namespace cv;
using namespace std;

FaceTracker tracker("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml", 0);

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  return -1; // check if we succeeded

    Mat frame;
    namedWindow("tracking",1);

    while(true)
    {
        cap >> frame;

        tracker.track(frame);

        for(auto face_center : tracker.faces_centers){
             ellipse( frame, face_center, Size( 5, 5), 0, 0, 360, Scalar( 255, 255, 0 ), 4, 8, 0 ); // could be boxe-rectangle ~ size of face
        }

        imshow("tracking", frame);
        if(waitKey(30) >= 0) break;	

    }

    return 0;
}








