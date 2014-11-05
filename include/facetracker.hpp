#ifndef FACETRACKER_HPP
#define FACETRACKER_HPP

#include <opencv2/opencv.hpp>

class FaceTracker
{
public:
    // constructor
    FaceTracker(const std::string& cascade_src, int update_rate); // ? const ??

    // public methods
    void track(cv::Mat&);

    // public variables
    cv::Mat _debug;
    std::vector<cv::Point2f> faces_centers;

private:
    // private methods
    void detecting(cv::Mat);
    void tracking(cv::InputArray frame);
    void update(std::vector<cv::Point2f>);
    std::vector<cv::Point2f> findCenters( std::vector<std::vector<cv::Point2f>> );
    std::vector<cv::Point2f> findGoodCenters( std::vector<std::vector<cv::Point2f>>, std::vector<cv::Point2f> );
    int findNewPoint(std::vector<cv::Point2f>, std::vector<cv::Point2f> );

    // private variables
    cv::Mat prev_frame;
    cv::CascadeClassifier face_cascade;
    int update_rate;
    int counter;
    std::vector<std::vector<cv::Point2f>> faces_features;
};

#endif // FACETRACKER_HPP
