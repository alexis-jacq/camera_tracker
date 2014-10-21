#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <facetracker.hpp>

using namespace cv;
using namespace std;



/** Global variables */
String face_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
FaceTracker tracker;



int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  return -1; // check if we succeeded

    //Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ cout << "--(!)Error loading"<< std::endl ; return -1; };


    // init : face detection
    //---------------------
    Mat frame;
    Mat gframe; // gray frame that will be used for goodfeaturtotrack
    cap >> frame; // get a new frame from camera
    cvtColor(frame, gframe, CV_BGR2GRAY); // color -> gray

    vector<vector<Point2f>> boxes = tracker.detect(gframe,face_cascade);


    // loop : face tracking
    //---------------------
    Mat next_gframe;
    int modul = 0;
    namedWindow("tracking",1);
    while(true)
    {
        // new frame processing
        cap >> frame;
        cvtColor(frame, next_gframe, CV_BGR2GRAY);

        vector<Point2f> goodCenters = tracker.track(gframe,next_gframe,boxes);

        for(auto center : goodCenters){
            ellipse( frame, center, Size( 5, 5), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
        }

        imshow("tracking", frame);
        if(waitKey(30) >= 0) break;	


        // every each 50 frame : check if there is a face just appears or disappears
        if(modul==50){
            tracker.update(gframe, boxes,goodCenters,face_cascade);
            modul=0;
        }
        else modul++;
    }

    return 0;
}








