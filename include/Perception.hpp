#ifndef PERCEPTION_HPP
#define PERCEPTION_HPP

#include <opencv2/opencv.hpp>

class Perception {
public:
    virtual ~Perception() {}
    virtual bool read() = 0;
    virtual bool getCamera(cv::Mat &sample) const = 0;
    virtual bool getIMU(std::array<float, 6> &sample) const = 0;
};

#endif
