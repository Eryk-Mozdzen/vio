#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>

#include "Dataset.hpp"

Dataset::Dataset(const std::string &path) : path{path} {
    {
        std::ifstream file(path + "/mav0/cam0/data.csv");
        std::string line;
        std::getline(file, line);
        while(std::getline(file, line)) {
            std::stringstream ss(line);
            std::string timestamp, filename;
            std::getline(ss, timestamp, ',');
            std::getline(ss, filename);
            camera.emplace_back(std::stod(timestamp) * 1E-9, filename);
        }
    }

    {
        std::ifstream file(path + "/mav0/imu0/data.csv");
        std::string line;
        std::getline(file, line);
        while(std::getline(file, line)) {
            std::stringstream ss(line);
            std::string timestamp, wx, wy, wz, ax, ay, az;
            std::getline(ss, timestamp, ',');
            std::getline(ss, wx, ',');
            std::getline(ss, wy, ',');
            std::getline(ss, wz, ',');
            std::getline(ss, ax, ',');
            std::getline(ss, ay, ',');
            std::getline(ss, az);
            imu.emplace_back(std::stod(timestamp) * 1E-9, std::array<float, 6>{
                                                              std::stof(wx),
                                                              std::stof(wy),
                                                              std::stof(wz),
                                                              std::stof(ax),
                                                              std::stof(ay),
                                                              std::stof(az),
                                                          });
        }
    }

    const double datasetStartTime = std::min(camera.at(0).first, imu.at(0).first);

    for(auto &sample : camera) {
        sample.first -= datasetStartTime;
    }

    for(auto &sample : imu) {
        sample.first -= datasetStartTime;
    }

    start = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();

    cameraSample = camera.begin();
    imuSample = imu.begin();

    which = 0;
}

bool Dataset::read() {
    if(which == 1) {
        cameraSample++;
    }

    if(which == 2) {
        imuSample++;
    }

    which = 0;

    if((cameraSample == camera.end()) || (imuSample == imu.end())) {
        return false;
    }

    const double now = std::chrono::duration_cast<std::chrono::duration<double>>(
                           std::chrono::high_resolution_clock::now().time_since_epoch())
                           .count();
    const double elapsed = now - start;

    if(cameraSample->first < imuSample->first) {
        const double sleep = cameraSample->first - elapsed;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleep));
        which = 1;
    } else {
        const double sleep = imuSample->first - elapsed;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleep));
        which = 2;
    }

    return true;
}

bool Dataset::getCamera(cv::Mat &sample) const {
    if(which != 1) {
        return false;
    }

    sample = cv::imread(path + "/mav0/cam0/data/" + cameraSample->second);
    return true;
}

bool Dataset::getIMU(std::array<float, 6> &sample) const {
    if(which != 2) {
        return false;
    }

    sample = imuSample->second;
    return true;
}
