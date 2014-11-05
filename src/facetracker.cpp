#include "facetracker.hpp"
#include <opencv2/opencv.hpp>

#define DEBUG

using namespace cv;
using namespace std;



// take prev and next image and boxes, update next boxes, and return centers of boxes
//-----------------------------------------------------------------------------------
void FaceTracker::track(Mat &frame) //, Mat next_init_frame, vector<vector<Point2f>> &boxes
{

    if(this->counter > 0){
        // case tracking
        //-------------
        this->tracking(frame);
    }
    else {
        // case detecting
        //---------------
        this->detecting(frame);

    }
}


// detect faces and set ood features to track
//-------------------------------------------
void FaceTracker::detecting(Mat frame){

    Mat init_frame;
    cvtColor(frame, init_frame, CV_BGR2GRAY);
    vector<Rect> faces_rect; // 2D vector for the list of faces

    this->face_cascade.detectMultiScale(init_frame, faces_rect, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );// detect faces in init_frame

     for(auto face_rect : faces_rect){

         // create the mask
         Point2f center = (face_rect.tl() + face_rect.br())*0.5;
         Mat mask(init_frame.rows,init_frame.cols, CV_8UC1, Scalar(0,0,0));
         ellipse(mask, center, Size(face_rect.width*0.2,face_rect.height*0.4),0,0,360,Scalar(255,255,255),-1,8);

         // get features in the mask
         vector<Point2f> corners;
         goodFeaturesToTrack(init_frame,corners,10,0.01,10,mask=mask,3,false,0.04); // get the good corners to track

         this->faces_features.push_back(corners);
         this->prev_frame = init_frame.clone();
     }
     vector<Point2f> centers = this->findCenters(this->faces_features);
     this->faces_centers = this->findGoodCenters( this->faces_features, centers);

     this->counter++;

}


// track features and update ~ update_rate
//----------------------------------------
void FaceTracker::tracking(InputArray frame){

#ifdef DEBUG
    //Mat _debug = frame.clone();
#endif

    Mat next_frame;
    cvtColor(frame, next_frame, CV_BGR2GRAY);

    vector<uchar> status;
    Mat err;

    vector<vector<Point2f>> new_faces_features;

    for(auto face_features : this->faces_features){
        vector<Point2f> next_face_features;
        calcOpticalFlowPyrLK(this->prev_frame,next_frame,face_features,next_face_features,status,err);
        new_faces_features.push_back(next_face_features);
    }

    this->faces_features = new_faces_features;

    this->prev_frame = next_frame.clone();

    vector<Point2f> centers = this->findCenters(this->faces_features);
    this->faces_centers = this->findGoodCenters( this->faces_features, centers);

#ifdef DEBUG
    //for(auto center : this->faces_centers){
        //ellipse( _debug, center, Size( 5, 5), 0, 0, 360, Scalar( 0, 255, 255 ), 4, 8, 0 ); // will be boxe-rectangle ~ size of face
    //}
#endif

    // update
    //-------
    if(this->update_rate == 0){ // default case

        if(this->counter > 50){

            this->update(this->faces_centers);
            this->counter = 1;
        }
        else this->counter++;
    }

    if(this->update_rate > 0){ // user case
        if(this->counter > this->update_rate){

            this->update(this->faces_centers);
            this->counter = 1;
        }
        else this->counter++;
    }

    // if his->update_rate < 0 : no-updtate case

}



// find the center of each set of corner in boxes
//-----------------------------------------------
vector<Point2f> FaceTracker::findCenters( vector<vector<Point2f>> boxes){

    vector<Point2f> centers;

    for(auto boxe : boxes){

        Point2f center(0,0);
        float i = 0;

        for(auto corner : boxe){
            center = center + corner;
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
void FaceTracker::update(vector<Point2f> centers){


    // detect
    //-------
    vector<Rect> faces_rect;
    this->face_cascade.detectMultiScale(this->prev_frame, faces_rect, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );// detect faces in init_frame
    vector<vector<Point2f>> faces_features_test;

     for(auto face_rect : faces_rect){

         // create the mask
         Point2f center = (face_rect.tl() + face_rect.br())*0.5;
         Mat mask(this->prev_frame.rows,this->prev_frame.cols, CV_8UC1, Scalar(0,0,0));
         ellipse(mask, center, Size(face_rect.width*0.2,face_rect.height*0.4),0,0,360,Scalar(255,255,255),-1,8);

         // get features in the mask
         vector<Point2f> corners;
         goodFeaturesToTrack(this->prev_frame,corners,10,0.01,10,mask=mask,3,false,0.04); // get the good corners to track

         faces_features_test.push_back(corners);
     }

     //compare
     //-------
    vector<Point2f> centersTest = this->findCenters(faces_features_test); // centers of the new faces
    vector<Point2f> goodCentersTest = this->findGoodCenters( faces_features_test, centersTest);//

    // if new faces appear :
    if(faces_features_test.size() > this->faces_features.size()){
        //cout << "new face detected"<< endl;
        int indice = this->findNewPoint(centers, goodCentersTest);
        this->faces_features.push_back(faces_features_test[indice]);
    }

    // if faces disappear
    if(faces_features_test.size() < this->faces_features.size()){
        //cout << "face lost"<< endl;
        int indice = this->findNewPoint(goodCentersTest, centers);
        vector<vector<Point2f>> new_Faces_features;
        if(indice>0){
            for(int i = 0; i<indice; i++) new_Faces_features.push_back(this->faces_features[i]);
        }
        if(indice+1 < this->faces_features.size()){
            for(int i = indice+1; i<this->faces_features.size(); i++) new_Faces_features.push_back(this->faces_features[i]);
        }
        this->faces_features = new_Faces_features;
        //cout<<this->faces_features.size()<<endl;
    }
}




// constructor
//------------
FaceTracker::FaceTracker(const string &cascade_src, int update_rate)
{
    CascadeClassifier face_cascade;
    if( !face_cascade.load( cascade_src ) ){ cout << "--(!)Error loading"<< std::endl ; };
    Mat prev_frame;
    vector<vector<Point2f>> faces_features;
    vector<Point2f> faces_centers;

    this->faces_centers = faces_centers;
    this->face_cascade = face_cascade;
    this->update_rate = update_rate;
    this->prev_frame = prev_frame;
    this->faces_features = faces_features;
    this->counter = 0;

}
