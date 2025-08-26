#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>

#include "MSCKF.hpp"

template <int N>
struct Vector : public Eigen::Vector<float, N> {
    Vector() : Eigen::Vector<float, N>(Eigen::Vector<float, N>::Zero()) {
    }

    template <typename Derived>
    Vector(const Eigen::MatrixBase<Derived> &other) : Eigen::Vector<float, N>(other) {
    }
};

class PropagationIMU {
    const Eigen::Vector3f wm;
    const Eigen::Vector3f am;

public:
    PropagationIMU(const Eigen::Vector3f wm, const Eigen::Vector3f am) : wm{wm}, am{am} {
    }

    void operator()(const Vector<16> &y, Vector<16> &dydt, double) const {
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
        const Eigen::Vector3f dv = C.transpose() * a + g;

        dydt.segment<4>(0) = dq;
        dydt.segment<3>(4) = Eigen::Vector3f::Zero();
        dydt.segment<3>(7) = dv;
        dydt.segment<3>(10) = Eigen::Vector3f::Zero();
        dydt.segment<3>(13) = v;
    }
};

class PropagationPii {
    const Eigen::Matrix<float, 15, 15> F;
    const Eigen::Matrix<float, 15, 12> G;
    const Eigen::Matrix<float, 12, 12> Qimu;

public:
    PropagationPii(const Eigen::Matrix<float, 15, 15> F, const Eigen::Matrix<float, 15, 12> G)
        : F{F}, G{G}, Qimu{Eigen::Matrix<float, 12, 12>::Identity() * 0.001f} {
    }

    void operator()(const Vector<15 * 15> &y, Vector<15 * 15> &dydt, double) const {
        Eigen::Matrix<float, 15, 15> Pii = Eigen::Matrix<float, 15, 15>::Zero();
        for(int r = 0; r < 15; r++) {
            for(int c = 0; c < 15; c++) {
                Pii(r, c) = y((15 * r) + c);
            }
        }

        const Eigen::Matrix<float, 15, 15> dPiidt =
            (F * Pii) + (Pii * F.transpose()) + (G * Qimu * G.transpose());

        for(int r = 0; r < 15; r++) {
            for(int c = 0; c < 15; c++) {
                dydt((15 * r) + c) = dPiidt(r, c);
            }
        }
    }
};

class PropagationPhi {
    const Eigen::Matrix<float, 15, 15> F;

public:
    PropagationPhi(const Eigen::Matrix<float, 15, 15> F) : F{F} {
    }

    void operator()(const Vector<15 * 15> &y, Vector<15 * 15> &dydt, double) const {
        Eigen::Matrix<float, 15, 15> Phi = Eigen::Matrix<float, 15, 15>::Zero();
        for(int r = 0; r < 15; r++) {
            for(int c = 0; c < 15; c++) {
                Phi(r, c) = y((15 * r) + c);
            }
        }

        const Eigen::Matrix<float, 15, 15> dPhidt = F * Phi;

        for(int r = 0; r < 15; r++) {
            for(int c = 0; c < 15; c++) {
                dydt((15 * r) + c) = dPhidt(r, c);
            }
        }
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

    P.resize(15 + (6 * N), 15 + (6 * N));
    P.setIdentity();
}

void MSCKF::propagate(const Eigen::Vector3f &gyro, const Eigen::Vector3f &accel) {
    PropagationIMU imu_system(gyro, accel);
    Vector<16> imu_state = x.segment<16>(0);

    boost::numeric::odeint::integrate_adaptive(
        boost::numeric::odeint::make_controlled(
            1E-6, 1E-6, boost::numeric::odeint::runge_kutta_dopri5<Vector<16>>()),
        imu_system, imu_state, static_cast<double>(0), static_cast<double>(T),
        static_cast<double>(0.01f * T));

    x.segment<16>(0) = imu_state;

    const Eigen::MatrixXf Piikk = P.block(0, 0, 15, 15);
    const Eigen::MatrixXf Pickk = P.block(0, 15, 15, 6 * N);

    const Eigen::Quaternionf q(x.segment<4>(0));
    const Eigen::Vector3f bg = x.segment<3>(4);
    const Eigen::Vector3f ba = x.segment<3>(12);

    const Eigen::Vector3f w = gyro - bg;
    const Eigen::Vector3f a = accel - ba;

    const Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    const Eigen::Matrix3f C = q.matrix();
    const Eigen::Matrix3f W{
        {0,     -w(2), w(1) },
        {w(2),  0,     -w(0)},
        {-w(1), w(0),  0    },
    };
    const Eigen::Matrix3f A{
        {0,     -a(2), a(1) },
        {a(2),  0,     -a(0)},
        {-a(1), a(0),  0    },
    };

    Eigen::Matrix<float, 15, 15> F = Eigen::Matrix<float, 15, 15>::Zero();
    F.block<3, 3>(0, 0) = -W;
    F.block<3, 3>(0, 3) = -I;
    F.block<3, 3>(6, 0) = -C.transpose() * A;
    F.block<3, 3>(6, 9) = -C.transpose();
    F.block<3, 3>(12, 6) = I;

    Eigen::Matrix<float, 15, 12> G = Eigen::Matrix<float, 15, 12>::Zero();
    G.block<3, 3>(0, 0) = -I;
    G.block<3, 3>(3, 3) = I;
    G.block<3, 3>(6, 6) = -C.transpose();
    G.block<3, 3>(9, 9) = I;

    PropagationPii Piik1k_system(F, G);
    Vector<15 * 15> Piik1k_state;
    for(int r = 0; r < 15; r++) {
        for(int c = 0; c < 15; c++) {
            Piik1k_state((15 * r) + c) = Piikk(r, c);
        }
    }

    boost::numeric::odeint::integrate_adaptive(
        boost::numeric::odeint::make_controlled(
            1E-6, 1E-6, boost::numeric::odeint::runge_kutta_dopri5<Vector<15 * 15>>()),
        Piik1k_system, Piik1k_state, static_cast<double>(0), static_cast<double>(T),
        static_cast<double>(0.01f * T));

    Eigen::Matrix<float, 15, 15> Piik1k = Eigen::Matrix<float, 15, 15>::Zero();
    for(int r = 0; r < 15; r++) {
        for(int c = 0; c < 15; c++) {
            Piik1k(r, c) = Piik1k_state((15 * r) + c);
        }
    }

    PropagationPhi phi_system(F);
    Eigen::Matrix<float, 15, 15> phi = Eigen::Matrix<float, 15, 15>::Identity();
    Vector<15 * 15> phi_state;
    for(int r = 0; r < 15; r++) {
        for(int c = 0; c < 15; c++) {
            phi_state((15 * r) + c) = phi(r, c);
        }
    }

    boost::numeric::odeint::integrate_adaptive(
        boost::numeric::odeint::make_controlled(
            1E-6, 1E-6, boost::numeric::odeint::runge_kutta_dopri5<Vector<15 * 15>>()),
        phi_system, phi_state, static_cast<double>(0), static_cast<double>(T),
        static_cast<double>(0.01f * T));

    for(int r = 0; r < 15; r++) {
        for(int c = 0; c < 15; c++) {
            phi(r, c) = phi_state((15 * r) + c);
        }
    }

    P.block(0, 0, 15, 15) = Piik1k;
    P.block(0, 15, 15, 6 * N) = phi * Pickk;
    P.block(15, 0, 6 * N, 15) = Pickk.transpose() * phi.transpose();
}

Eigen::Quaternionf MSCKF::getOrientation() const {
    return Eigen::Quaternionf(x.segment<4>(0));
}

Eigen::Vector3f MSCKF::getPosition() const {
    return x.segment<3>(12);
}
