#include "facetracker.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;



// take prev and next image and boxes, update next boxes, and return centers of boxes
//-----------------------------------------------------------------------------------
vector<Point2f> FaceTracker::track(Mat &gframe, Mat next_gframe, vector<vector<Point2f>> &boxes)
{
    vector<vector<Point2f>> next_boxes;
    this->trackBoxes(gframe,next_gframe,boxes,next_boxes); // tracking
    boxes = next_boxes;
    gframe = next_gframe.clone();
    vector<Point2f> centers = this->findCenters(next_boxes);
    vector<Point2f> goodCenters = this->findGoodCenters( next_boxes, centers);

    return goodCenters;
}



// track next points for each boxes of a gray-frame gframe in a next gray-frame next_gframe
//-----------------------------------------------------------------------------------------
void FaceTracker::trackBoxes(Mat gframe, Mat next_gframe, vector<vector<Point2f>> boxes, vector<vector<Point2f>> &next_boxes){

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
vector<Point2f> FaceTracker::findCenters( vector<vector<Point2f>> boxes){

    vector<Point2f> centers;

    for(auto boxe : boxes){

                        vector<double> calcGravities( vector<vector<Point2f>>, vector<Point2f>);
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




// find more robust centers : square preference for points close to the centers
//-----------------------------------------------------------------------------
vector<Point2f> FaceTracker::findGoodCenters( vector<vector<Point2f>> boxes, vector<Point2f> centers){

    vector<Point2f> goodCenters;

    int index = 0;
    for(auto boxe : boxes){

        Point2f center = centers[index];
        Point2f goodCenter(0,0);
        double i = 0;

        for(auto corner : boxe){
            double delta = norm(center - corner);
            double omega = 1./(delta*delta);
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


// detects faces in a gray frame and returns for each face the set of best points to track
//----------------------------------------------------------------------------------------
vector<vector<Point2f>> FaceTracker::detect(Mat gframe, CascadeClassifier face_cascade){

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



// find the indice of the new detected face : this is the "most alone' center
//---------------------------------------------------------------------------
int FaceTracker::findNewPoint(vector<Point2f> centers, vector<Point2f> tests){

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




// check if there is a face just appears or disappears
//----------------------------------------------------
void  FaceTracker::update(Mat gframe, vector<vector<Point2f>>& boxes, vector<Point2f> centers, CascadeClassifier face_cascade){

    vector<vector<Point2f>> boxesTest = this->detect(gframe,face_cascade); // new faces detection to compare
    vector<Point2f> centersTest = this->findCenters(boxesTest); // centers of the new faces
    vector<Point2f> goodCentersTest = this->findGoodCenters( boxesTest, centersTest);//

    // if new faces appear :
    if(boxesTest.size() > boxes.size()){
        cout << "new face detected"<< endl;
        int indice = this->findNewPoint(centers, goodCentersTest);
        boxes.push_back(boxesTest[indice]);
    }

    // if faces disappear
    if(boxesTest.size() < boxes.size()){
        cout << "face lost"<< endl;
        int indice = this->findNewPoint(goodCentersTest, centers);
        vector<vector<Point2f>> newBoxes;
        if(indice>0){
            for(int i = 0; i<indice; i++) newBoxes.push_back(boxes[i]);
        }
        if(indice+1<boxes.size()){
            for(int i = indice+1; i<boxes.size(); i++) newBoxes.push_back(boxes[i]);
        }
        boxes = newBoxes;
        cout<<newBoxes.size()<<endl;
    }
}
