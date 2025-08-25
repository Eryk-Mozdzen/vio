#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>

#include "MSCKF.hpp"

class PropagationModel {
    const Eigen::Vector3f wm;
    const Eigen::Vector3f am;

public:
    using State = Eigen::Vector<float, 16>;
    using Stepper = boost::numeric::odeint::runge_kutta_dopri5<State>;

    PropagationModel(const Eigen::Vector3f gyro, const Eigen::Vector3f accel)
        : wm{gyro}, am{accel} {
    }

    void operator()(const State &y, State &dydt, double) const {
        const Eigen::Quaternionf q(y.segment<4>(0));
        const Eigen::Vector3f bg = y.segment<3>(4);
        const Eigen::Vector3f v = y.segment<3>(7);
        const Eigen::Vector3f ba = y.segment<3>(10);

        const Eigen::Vector3f g(0, 0, -9.8065f);
        const Eigen::Vector3f w = wm - bg;
        const Eigen::Vector3f a = am - ba;
        const Eigen::Matrix3f C = q.matrix();
        const Eigen::Matrix4f Ow{
            {0,     -w(2), w(1),  w(0)},
            {w(2),  0,     -w(0), w(1)},
            {-w(1), w(0),  0,     w(2)},
            {-w(0), -w(1), -w(2), 0   },
        };

        const Eigen::Vector4f dq = 0.5f * Ow * q.coeffs();
        const Eigen::Vector3f dv = C.transpose() * a + C * g;

        dydt.segment(0, 4) = dq;
        dydt.segment(4, 3) = Eigen::Vector3f::Zero();
        dydt.segment(7, 3) = dv;
        dydt.segment(10, 3) = Eigen::Vector3f::Zero();
        dydt.segment(13, 3) = v;
    }
};

MSCKF::MSCKF(const int cameraPoses, const float imuSamplePeriod)
    : N{cameraPoses}, T{imuSamplePeriod} {
    x.resize(16 + 7 * N);
    x.segment<4>(0) = Eigen::Quaternionf::Identity().coeffs();
    x.segment<3>(4) = Eigen::Vector3f::Zero();
    x.segment<3>(7) = Eigen::Vector3f::Zero();
    x.segment<3>(10) = Eigen::Vector3f::Zero();
    x.segment<3>(13) = Eigen::Vector3f::Zero();

    for(int i = 0; i < N; i++) {
        x.segment<4>(16 + (7 * i) + 0) = Eigen::Quaternionf::Identity().coeffs();
        x.segment<3>(16 + (7 * i) + 4) = Eigen::Vector3f::Zero();
    }

    P.resize(16 + (7 * N), 16 + (7 * N));
    P.setIdentity();
}

void MSCKF::propagate(const Eigen::Vector3f &gyro, const Eigen::Vector3f &accel) {
    PropagationModel system(gyro, accel);
    PropagationModel::State state = x.segment<16>(0);

    boost::numeric::odeint::integrate_adaptive(
        boost::numeric::odeint::make_controlled(1E-12, 1E-12, PropagationModel::Stepper()), system,
        state, 0., 1. * T, 0.1 * T);

    x.segment<16>(0) = state;
}

const Eigen::VectorXf &MSCKF::get() const {
    return x;
}
