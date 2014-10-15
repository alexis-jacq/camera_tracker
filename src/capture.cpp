#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/** Global variables */
String face_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera

    if(!cap.isOpened())  // check if we succeeded
    {
        return -1;
    }

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ cout << "--(!)Error loading"<< std::endl ; return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ cout << "--(!)Error loading"<< std::endl ; return -1; };

    // init features with goodfeaturtotrack
    //-------------------------------------
    Mat frame;
    Mat gframe; // gray frame that will be used for goodfeaturtotrack
    cap >> frame; // get a new frame from camera
    cvtColor(frame, gframe, CV_BGR2GRAY); // color -> gray
    vector<Point2f> pts; // 2D vector for the list of corner points
    goodFeaturesToTrack(gframe,pts,30,0.01,10,noArray(),3,false,0.04); // get the good corners

    // loop with tracking with calcOptimalFlow
    //----------------------------------------
    namedWindow("tracking",1);

    Mat next_gframe;
    vector<Point2f> next_pts;
    Vec3b color;
    std::vector<Rect> faces;

    while(true)
    {
        cap >> frame;
        cvtColor(frame, next_gframe, CV_BGR2GRAY);

        vector<uchar> status;
        Mat err;

        calcOpticalFlowPyrLK(gframe,next_gframe,pts,next_pts,status,err);//,Size(21,21),3,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),0,1e-4); // compute next positions of features

        for(int i = 0; i < next_pts.size(); i++){     // change the color of the found corners
            color = frame.at<Vec3b>(next_pts[i]); // get color-pixel
            color[1]=255;
            color[2]=0;
            color[3]=0;
            frame.at<Vec3b>(next_pts[i]) = color; // set color
        }

        // face cascade detection
        //-----------------------

        // detect + display faces and eyes
        face_cascade.detectMultiScale( next_gframe, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        for( size_t i = 0; i < faces.size(); i++ )
        {
            Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
            ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
         }


        imshow("tracking", frame);

        if(waitKey(30) >= 0) break;

        gframe = next_gframe.clone();
        pts = next_pts;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor

    return 0;
}
