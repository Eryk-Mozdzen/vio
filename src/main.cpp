#include <opencv2/opencv.hpp>

#include "Client.hpp"
#include "Dataset.hpp"
#include "MSCKF.hpp"
#include "Perception.hpp"

#define FEATURES 15

void undistort(cv::Mat src, cv::Mat &dst) {
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    K.at<double>(0, 0) = 190.97847715128717;
    K.at<double>(1, 1) = 190.9733070521226;
    K.at<double>(0, 2) = 254.93170605935475;
    K.at<double>(1, 2) = 256.8974428996504;
    K.at<double>(2, 2) = 1;

    cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
    D.at<double>(0, 0) = 0.0034823894022493434;
    D.at<double>(1, 0) = 0.0007150348452162257;
    D.at<double>(2, 0) = -0.0020532361418706202;
    D.at<double>(3, 0) = 0.00020293673591811182;

    cv::Mat map1;
    cv::Mat map2;
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_64F), K, cv::Size(512, 512),
                                         CV_32FC1, map1, map2);
    cv::remap(src, dst, map1, map2, cv::INTER_CUBIC, cv::BORDER_CONSTANT);
}

int main() {
    Client client;
    client.write("{\"command\" : \"clear\"}");
    client.write("{\"command\" : \"config\", \"theme\" : \"dark\", \"camera\" : \"orbit\"}");
    client.write("{ \
        \"command\" : \"create\", \
        \"path\" : \"obj1\", \
        \"geometry\" : {\"shape\" : \"cuboid\", \"size\" : [ 0.25, 0.25, 0.25 ]}, \
        \"material\" : {\"color\" : [ 255, 0, 255 ]}, \
        \"transform\" : {\"translation\" : [ 0, 0, 1 ]} \
    }");

    MSCKF msckf(30, 0.01);

    Perception *perception = new Dataset("../datasets/dataset-corridor1_512_16");

    cv::Mat prevImage;
    std::vector<cv::Point2f> prevPoints;
    std::vector<int> ids;
    int counter = 0;

    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(50);

    cv::namedWindow("window", cv::WINDOW_AUTOSIZE);

    while(perception->read()) {
        cv::Mat input;
        if(perception->getCamera(input)) {
            cv::Mat gray;
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
            cv::Mat equalized;
            cv::equalizeHist(gray, equalized);
            undistort(equalized, prevImage);

            std::vector<cv::KeyPoint> keypoints;
            detector->detect(prevImage, keypoints);
            cv::KeyPoint::convert(keypoints, prevPoints);
            prevPoints.resize(FEATURES);
            ids.resize(prevPoints.size());
            for(size_t i = 0; i < prevPoints.size(); i++) {
                ids[i] = counter;
                counter++;
            }
            break;
        }
    }

    while(perception->read()) {
        cv::Mat input;
        if(perception->getCamera(input)) {
            cv::Mat gray;
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

            cv::Mat equalized;
            cv::equalizeHist(gray, equalized);

            cv::Mat image;
            undistort(equalized, image);

            std::vector<cv::Point2f> points;
            std::vector<unsigned char> status;
            std::vector<float> error;
            cv::calcOpticalFlowPyrLK(prevImage, image, prevPoints, points, status, error);

            std::vector<cv::KeyPoint> keypoints;
            detector->detect(image, keypoints);

            for(size_t i = 0; i < points.size(); i++) {
                if(!status[i]) {
                    double max_min_distance = -1;
                    for(const cv::KeyPoint &keypoint : keypoints) {
                        double min_distance = 1E+9;
                        for(size_t j = 0; j < points.size(); j++) {
                            if(status[j]) {
                                const double d = cv::norm(keypoint.pt - points[j]);
                                if(d < min_distance) {
                                    min_distance = d;
                                }
                            }
                        }
                        if(min_distance > max_min_distance) {
                            max_min_distance = min_distance;
                            points[i] = keypoint.pt;
                            status[i] = 1;
                            ids[i] = counter;
                        }
                    }
                    counter++;
                }
            }

            cv::Mat window;
            cv::cvtColor(image, window, cv::COLOR_GRAY2BGR);
            // cv::Mat window(image.size(), CV_8UC3, cv::Scalar());

            for(size_t i = 0; i < points.size(); i++) {
                cv::circle(window, points[i], 2, cv::Scalar(0, 255, 0), -1);
                cv::putText(window, std::to_string(ids[i]), points[i], cv::FONT_HERSHEY_SIMPLEX,
                            0.4, cv::Scalar(0, 0, 255));
            }

            cv::imshow("window", window);
            cv::waitKey(1);

            prevImage = image.clone();
            prevPoints = points;

            const Eigen::Quaternionf q = msckf.getOrientation();
            // const Eigen::Vector3f p = msckf.getPosition();

            std::stringstream ss;
            ss << "{";
            ss << "\"command\" : \"update\",";
            ss << "\"path\" : \"obj1\",";
            ss << "\"transform\" : {";
            // ss << "\"translation\" : [";
            // ss << p(0) << ", ";
            // ss << p(1) << ", ";
            // ss << p(2) << "],";
            ss << "\"quaternion\" : [";
            ss << q.w() << ", ";
            ss << q.x() << ", ";
            ss << q.y() << ", ";
            ss << q.z() << "]";
            ss << "}";
            ss << "}";
            client.write(ss.str());
        }

        std::array<float, 6> imu;
        if(perception->getIMU(imu)) {
            const Eigen::Vector3f gyro(imu[0], imu[1], imu[2]);
            const Eigen::Vector3f accel(imu[3], imu[4], imu[5]);
            msckf.propagate(gyro, accel);
        }
    }
}
