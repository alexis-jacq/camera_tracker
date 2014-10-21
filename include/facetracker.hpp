#ifndef FACETRACKER_HPP
#define FACETRACKER_HPP

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class FaceTracker
{
public:
    vector<Point2f> track(Mat&, Mat , vector<vector<Point2f>>&);
    vector<vector<Point2f>> detect(Mat ,  CascadeClassifier);
    void update(Mat, vector<vector<Point2f>>& ,vector<Point2f>, CascadeClassifier);

private:
    void trackBoxes(Mat, Mat , vector<vector<Point2f>> , vector<vector<Point2f>>&);
    vector<Point2f> findCenters( vector<vector<Point2f>> );
    vector<Point2f> findGoodCenters( vector<vector<Point2f>>, vector<Point2f> );
    int findNewPoint(vector<Point2f>, vector<Point2f> );

};

#endif // FACETRACKER_HPP
