#ifndef MSCKF_HPP
#define MSCKF_HPP

#include <Eigen/Dense>

class MSCKF {
    const int N;
    const float T;

    Eigen::VectorXf x;
    Eigen::MatrixXf P;

public:
    MSCKF(const int cameraPoses, const float imuSamplePeriod);

    void propagate(const Eigen::Vector3f &gyro, const Eigen::Vector3f &accel);
    void update(const std::vector<std::pair<Eigen::Vector2f, std::vector<int>>> &features);

    Eigen::Quaternionf getOrientation() const;
    Eigen::Vector3f getPosition() const;
};

#endif
