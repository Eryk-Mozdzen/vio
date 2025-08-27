#ifndef MSCKF_HPP
#define MSCKF_HPP

#include <map>
#include <vector>

#include <Eigen/Dense>

class MSCKF {
    const unsigned int N;
    const float T;

    Eigen::VectorXf x;
    Eigen::MatrixXf P;

    std::map<int, std::vector<Eigen::Vector2f>> features;

public:
    MSCKF(const unsigned int cameraPoses, const float imuSamplePeriod);

    void propagate(const Eigen::Vector3f &gyro, const Eigen::Vector3f &accel);
    void update(const std::vector<int> &ids, const std::vector<Eigen::Vector2f> &points);

    Eigen::Quaternionf getOrientation() const;
    Eigen::Vector3f getPosition() const;
};

#endif
