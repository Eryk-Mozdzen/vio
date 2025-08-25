#ifndef DATASET_HPP
#define DATASET_HPP

#include <chrono>

#include "Perception.hpp"

class Dataset : public Perception {
    const std::string path;
    std::vector<std::pair<double, std::string>> camera;
    std::vector<std::pair<double, std::array<float, 6>>> imu;
    double start;

    std::vector<std::pair<double, std::string>>::iterator cameraSample;
    std::vector<std::pair<double, std::array<float, 6>>>::iterator imuSample;

    int which;

public:
    Dataset(const std::string &path);

    bool read() override;
    bool getCamera(cv::Mat &sample) const override;
    bool getIMU(std::array<float, 6> &sample) const override;
};

#endif
