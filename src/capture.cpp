#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/** Functions headers */
vector<vector<Point2f>> facePointsToTrack(Mat );
void trackBoxes(Mat , Mat , vector<vector<Point2f>> , vector<vector<Point2f>>& ) ;
vector<Point2f> findCenters( vector<vector<Point2f>> ) ;
vector<Point2f> findGoodCenters( vector<vector<Point2f>> , vector<Point2f> );
int findNewPoint(vector<Point2f>, vector<Point2f> );

/** Global variables */
String face_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_eye.xml";
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


    // init : face detection
    //---------------
    Mat frame;
    Mat gframe; // gray frame that will be used for goodfeaturtotrack
    cap >> frame; // get a new frame from camera
    cvtColor(frame, gframe, CV_BGR2GRAY); // color -> gray
    vector<vector<Point2f>> boxes = facePointsToTrack(gframe); //
    auto nb_faces = boxes.size();

    // loop : calcOpticalFlow tracking
    //----------------------------------------
    namedWindow("tracking",1);
    Mat next_gframe;
    int modul = 0;
    while(true)
    {
        cap >> frame;
        cvtColor(frame, next_gframe, CV_BGR2GRAY);
        vector<vector<Point2f>> next_boxes;
        trackBoxes(gframe,next_gframe,boxes,next_boxes); // tracking

        for(auto boxe : next_boxes){
            for(auto corner : boxe){
                ellipse( frame, corner, Size( 5, 5), 0, 0, 360, Scalar( 255, 255, 0 ), 4, 8, 0 );
            }
        }

        vector<Point2f> centers = findCenters(next_boxes);
        vector<Point2f> goodCenters = findGoodCenters( next_boxes, centers);

        //for(auto center : centers){
        //    ellipse( frame, center, Size( 5, 5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        //}
        for(auto goodCenter : goodCenters){
            ellipse( frame, goodCenter, Size( 5, 5), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 );
        }

        imshow("tracking", frame);

        if(waitKey(30) >= 0) break;

        gframe = next_gframe.clone();
        boxes = next_boxes;
        centers = goodCenters;


        if(modul==100){
            // check if new face
            vector<vector<Point2f>> boxesTest = facePointsToTrack(gframe); //
            vector<Point2f> centersTest = findCenters(boxesTest);
            auto nb_facesTest = boxesTest.size();

            if(nb_facesTest > nb_faces){
                cout << "new face detected"<< endl;
                nb_faces = nb_facesTest;
                int indice = findNewPoint(centers, centersTest);
                boxes.push_back(boxesTest[indice]);
            }

            if(nb_facesTest < nb_faces){
                cout << "face lost"<< endl;
                nb_faces = nb_facesTest;
                int indice = findNewPoint(centersTest, centers);
                vector<vector<Point2f>> newBoxes;
                if(indice>0){
                    for(int i = 0; i<indice; i++) newBoxes.push_back(boxes[i]);
                    if(indice+1<boxes.size()){
                        for(int i = indice+1; i<boxes.size(); i++) newBoxes.push_back(boxes[i]);
                    }
                }
                boxes = newBoxes;
            }
            modul=0;
        }
        else modul++;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor

    return 0;
}



// detects faces in a gray frame and returns for each face the set of best points to track
//----------------------------------------------------------------------------------------
vector<vector<Point2f>> facePointsToTrack(Mat gframe){

    vector<vector<Point2f>> boxes; // list of face-boxes with points to track

    vector<Rect> faces; // 2D vector for the list of faces
    face_cascade.detectMultiScale( gframe, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );// detect faces in gframe

     for(auto face : faces){

         // create the mask
         Point2f center = (face.tl() + face.br())*0.5;
         Mat mask(gframe.rows,gframe.cols, CV_8UC1, Scalar(0,0,0));
         ellipse(mask, center, Size(face.width*0.2,face.height*0.4),0,0,360,Scalar(255,255,255),-1,8);

         // get features in the mask
         vector<Point2f> corners;
         goodFeaturesToTrack(gframe,corners,10,0.01,10,mask=mask,3,false,0.04); // get the good corners

         boxes.push_back(corners);
     }

     return boxes;
}



// track next points for each boxes of gray frame gframe in next frame next_frame
//-------------------------------------------------------------------------------
void trackBoxes(Mat gframe, Mat next_gframe, vector<vector<Point2f>> boxes, vector<vector<Point2f>> &next_boxes){

    vector<uchar> status;
    Mat err;

    for(auto boxe : boxes){
        vector<Point2f> next_boxe;
        calcOpticalFlowPyrLK(gframe,next_gframe,boxe,next_boxe,status,err);

        next_boxes.push_back(next_boxe);
    }
}



// find the center of each set of corner in boxes
//-----------------------------------------------
vector<Point2f> findCenters( vector<vector<Point2f>> boxes){

    vector<Point2f> centers;

    for(auto boxe : boxes){

        Point2f center(0,0);
        float i = 0;

        for(auto corner : boxe){
            center = center + corner;
            //center = corner;
            i=i+1.;
        }
        if(i>0) i = 1./i;
        if(i==0) i = 1.;
        center = center*i;
        centers.push_back(center);
    }

    return centers;
}


// find more robust centers
//-------------------------
vector<Point2f> findGoodCenters( vector<vector<Point2f>> boxes, vector<Point2f> centers){

    vector<Point2f> goodCenters;

    int index = 0;
    for(auto boxe : boxes){

        Point2f center = centers[index];
        Point2f goodCenter(0,0);
        float i = 0;

        for(auto corner : boxe){
            float delta = norm(center - corner);
            float omega = 1./(delta*delta);
            goodCenter = goodCenter + omega*corner;
            i=i+omega;
        }
        if(i>0) i = 1./i;
        if(i==0) i = 1.;
        goodCenter = goodCenter*i;
        goodCenters.push_back(goodCenter);

        index++;
    }

    return goodCenters;
}

// find the indice of the new detected face
//-----------------------------------------
int findNewPoint(vector<Point2f> centers, vector<Point2f> tests){

    int index_max = 0;
    double d_max = 0;

    if(centers.size()>0){
        for(int i=0; i<tests.size();i++){

            double d_min = 100000;
            int index_min = 0;

            for(int j=0; j<centers.size();j++){

                if(norm(tests[i] - centers[j]) < d_min ){
                    d_min = norm(tests[i] - centers[j]);
                    index_min = j;
                }
            }
            if(norm(tests[i] - centers[index_min]) > d_max ){
                d_max = norm(tests[i] - centers[index_min]);
                index_max = i;
            }
        }
    }

    return index_max;
}

