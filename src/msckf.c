#include <gsl/gsl_blas.h>
#include <gsl/gsl_odeiv2.h>

#include "msckf.h"

#define N              30
#define IMU_T          0.01
#define IMU_COVARIANCE 0.01

#define CAM_IMU_QX 0
#define CAM_IMU_QY -0.7071068
#define CAM_IMU_QZ 0.7071068
#define CAM_IMU_QW 0

#define CAM_IMU_PX 0.05
#define CAM_IMU_PY -0.07
#define CAM_IMU_PZ -0.05

typedef struct {
    double x;
    double y;
} frame_t;

typedef struct feature {
    struct feature *next;
    unsigned int id;
    int seen;
    frame_t frames[N];
    size_t count;
} feature_t;

struct msckf {
    gsl_vector *x;
    gsl_matrix *P;
    feature_t *features;
};

static feature_t *feature_find(feature_t *head, const unsigned int id) {
    feature_t *f = head;
    while(f) {
        if(f->id == id) {
            return f;
        }
        f = f->next;
    }
    return NULL;
}

static feature_t *feature_add(feature_t **head, unsigned int id) {
    feature_t *f = malloc(sizeof(feature_t));
    f->id = id;
    f->count = 0;
    f->next = *head;
    *head = f;
    return f;
}

static void feature_add_frame(feature_t *f, const double x, const double y) {
    if(f->count < N) {
        f->count++;
    }

    for(size_t i = f->count - 1; i >= 1; i--) {
        f->frames[i] = f->frames[i - 1];
    }

    f->frames[0].x = x;
    f->frames[0].y = y;
}

static void cross_matrix(const gsl_vector *vec, gsl_matrix *mat) {
    const double x = gsl_vector_get(vec, 0);
    const double y = gsl_vector_get(vec, 1);
    const double z = gsl_vector_get(vec, 2);

    gsl_matrix_set(mat, 0, 0, 0);
    gsl_matrix_set(mat, 0, 1, -z);
    gsl_matrix_set(mat, 0, 2, y);

    gsl_matrix_set(mat, 1, 0, z);
    gsl_matrix_set(mat, 1, 1, 0);
    gsl_matrix_set(mat, 1, 2, -x);

    gsl_matrix_set(mat, 2, 0, -y);
    gsl_matrix_set(mat, 2, 1, x);
    gsl_matrix_set(mat, 2, 2, 0);
}

static void rotation_matrix(const gsl_vector *quat, gsl_matrix *mat) {
    const double qx = gsl_vector_get(quat, 0);
    const double qy = gsl_vector_get(quat, 1);
    const double qz = gsl_vector_get(quat, 2);
    const double qw = gsl_vector_get(quat, 3);

    gsl_matrix_set(mat, 0, 0, 1 - (2 * qy * qy) - (2 * qz * qz));
    gsl_matrix_set(mat, 0, 1, (2 * qx * qy) - (2 * qz * qw));
    gsl_matrix_set(mat, 0, 2, (2 * qx * qz) + (2 * qy * qw));

    gsl_matrix_set(mat, 1, 0, (2 * qx * qy) + (2 * qz * qw));
    gsl_matrix_set(mat, 1, 1, 1 - (2 * qx * qx) - (2 * qz * qz));
    gsl_matrix_set(mat, 1, 2, (2 * qy * qz) - (2 * qx * qw));

    gsl_matrix_set(mat, 2, 0, (2 * qx * qz) - (2 * qy * qw));
    gsl_matrix_set(mat, 2, 1, (2 * qy * qz) + (2 * qx * qw));
    gsl_matrix_set(mat, 2, 2, 1 - (2 * qx * qx) - (2 * qy * qy));
}

static void quaternion_product(const gsl_vector *quat1, const gsl_vector *quat2, gsl_vector *quat) {
    const double q1x = gsl_vector_get(quat1, 0);
    const double q1y = gsl_vector_get(quat1, 1);
    const double q1z = gsl_vector_get(quat1, 2);
    const double q1w = gsl_vector_get(quat1, 3);

    const double q2x = gsl_vector_get(quat2, 0);
    const double q2y = gsl_vector_get(quat2, 1);
    const double q2z = gsl_vector_get(quat2, 2);
    const double q2w = gsl_vector_get(quat2, 3);

    gsl_vector_set(quat, 0, (q1w * q2x) + (q1x * q2w) + (q1y * q2z) - (q1z * q2y));
    gsl_vector_set(quat, 1, (q1w * q2y) - (q1x * q2z) + (q1y * q2w) + (q1z * q2x));
    gsl_vector_set(quat, 2, (q1w * q2z) + (q1x * q2y) - (q1y * q2x) + (q1z * q2w));
    gsl_vector_set(quat, 3, (q1w * q2w) - (q1x * q2x) - (q1y * q2y) - (q1z * q2z));
}

static int system_propagate_imu(double t, const double y[], double dydt[], void *params) {
    (void)t;

    gsl_vector_const_view q = gsl_vector_const_view_array(&y[0], 4);
    gsl_vector_const_view v = gsl_vector_const_view_array(&y[7], 3);

    gsl_vector_view *imu_params = params;
    gsl_vector_view w = imu_params[0];
    gsl_vector_view a = imu_params[1];

    double omega_data[4 * 4];
    gsl_matrix_view omega = gsl_matrix_view_array(omega_data, 4, 4);
    gsl_matrix_view omega_cross = gsl_matrix_submatrix(&omega.matrix, 0, 0, 3, 3);
    gsl_vector_view omega_vert = gsl_matrix_subcolumn(&omega.matrix, 3, 0, 3);
    gsl_vector_view omega_horz = gsl_matrix_subrow(&omega.matrix, 3, 0, 3);
    gsl_matrix_set_zero(&omega.matrix);
    cross_matrix(&w.vector, &omega_cross.matrix);
    gsl_matrix_scale(&omega_cross.matrix, -1);
    gsl_vector_memcpy(&omega_vert.vector, &w.vector);
    gsl_vector_memcpy(&omega_horz.vector, &w.vector);
    gsl_vector_scale(&omega_horz.vector, -1);

    double c_data[3 * 3];
    gsl_matrix_view c = gsl_matrix_view_array(c_data, 3, 3);
    rotation_matrix(&q.vector, &c.matrix);

    const double g_data[3] = {0, 0, -9.8065};
    gsl_vector_const_view g = gsl_vector_const_view_array(g_data, 3);

    gsl_vector_view dq = gsl_vector_view_array(&dydt[0], 4);
    gsl_vector_view dbg = gsl_vector_view_array(&dydt[4], 3);
    gsl_vector_view dv = gsl_vector_view_array(&dydt[7], 3);
    gsl_vector_view dba = gsl_vector_view_array(&dydt[10], 3);
    gsl_vector_view dp = gsl_vector_view_array(&dydt[13], 3);

    gsl_blas_dgemv(CblasNoTrans, 0.5, &omega.matrix, &q.vector, 0, &dq.vector);
    gsl_vector_set_zero(&dbg.vector);
    gsl_blas_dgemv(CblasTrans, 1, &c.matrix, &a.vector, 0, &dv.vector);
    gsl_vector_add(&dv.vector, &g.vector);
    gsl_vector_set_zero(&dba.vector);
    gsl_vector_memcpy(&dp.vector, &v.vector);

    return GSL_SUCCESS;
}

static int system_propagate_pii(double t, const double y[], double dydt[], void *params) {
    (void)t;

    gsl_matrix_const_view *pii_params = params;
    gsl_matrix_const_view f = pii_params[0];
    gsl_matrix_const_view g = pii_params[1];
    gsl_matrix_const_view pii = gsl_matrix_const_view_array(y, 15, 15);

    double qimu_data[12 * 12];
    gsl_matrix_view qimu = gsl_matrix_view_array(qimu_data, 12, 12);
    gsl_matrix_set_identity(&qimu.matrix);
    gsl_matrix_scale(&qimu.matrix, IMU_COVARIANCE);

    double gqimu_data[15 * 12];
    gsl_matrix_view gqimu = gsl_matrix_view_array(gqimu_data, 15, 12);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &g.matrix, &qimu.matrix, 0, &gqimu.matrix);

    gsl_matrix_view dpii = gsl_matrix_view_array(dydt, 15, 15);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &f.matrix, &pii.matrix, 0, &dpii.matrix);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, &pii.matrix, &f.matrix, 1, &dpii.matrix);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, &gqimu.matrix, &g.matrix, 1, &dpii.matrix);

    return GSL_SUCCESS;
}

static int system_propagate_phi(double t, const double y[], double dydt[], void *params) {
    (void)t;

    gsl_matrix_const_view *f = params;
    gsl_matrix_const_view phi = gsl_matrix_const_view_array(y, 15, 15);

    gsl_matrix_view dphi = gsl_matrix_view_array(dydt, 15, 15);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &f->matrix, &phi.matrix, 0, &dphi.matrix);

    return GSL_SUCCESS;
}

static void intagrate(int (*func)(double, const double[], double[], void *),
                      double state[],
                      size_t dim,
                      double time,
                      void *params) {
    const gsl_odeiv2_system system = {
        .function = func,
        .jacobian = NULL,
        .dimension = dim,
        .params = params,
    };

    gsl_odeiv2_step *stepper = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rkf45, dim);
    gsl_odeiv2_control *control = gsl_odeiv2_control_standard_new(1E-6, 1E-6, 1, 1);
    gsl_odeiv2_evolve *evolve = gsl_odeiv2_evolve_alloc(dim);

    double t = 0;
    double h = 1E-6;
    int status;

    while(t < time) {
        status = gsl_odeiv2_evolve_apply(evolve, control, stepper, &system, &t, time, &h, state);

        if(status != GSL_SUCCESS) {
            break;
        }
    }

    gsl_odeiv2_evolve_free(evolve);
    gsl_odeiv2_control_free(control);
    gsl_odeiv2_step_free(stepper);
}

msckf_t *msckf_create() {
    msckf_t *msckf = malloc(sizeof(msckf_t));

    msckf->x = gsl_vector_alloc(16 + (7 * N));
    msckf->P = gsl_matrix_alloc(15 + (6 * N), 15 + (6 * N));
    msckf->features = NULL;

    gsl_vector_view imu_q = gsl_vector_subvector(msckf->x, 0, 4);
    gsl_vector_view imu_bg = gsl_vector_subvector(msckf->x, 4, 3);
    gsl_vector_view imu_v = gsl_vector_subvector(msckf->x, 7, 3);
    gsl_vector_view imu_ba = gsl_vector_subvector(msckf->x, 10, 3);
    gsl_vector_view imu_p = gsl_vector_subvector(msckf->x, 13, 3);

    gsl_vector_set(&imu_q.vector, 0, 0);
    gsl_vector_set(&imu_q.vector, 1, 0);
    gsl_vector_set(&imu_q.vector, 2, 0);
    gsl_vector_set(&imu_q.vector, 3, 1);
    gsl_vector_set_zero(&imu_bg.vector);
    gsl_vector_set_zero(&imu_v.vector);
    gsl_vector_set_zero(&imu_ba.vector);
    gsl_vector_set_zero(&imu_p.vector);

    for(size_t i = 0; i < N; i++) {
        gsl_vector_view camera_q = gsl_vector_subvector(msckf->x, 16 + (7 * i), 4);
        gsl_vector_view camera_p = gsl_vector_subvector(msckf->x, 16 + (7 * i), 3);

        gsl_vector_set(&camera_q.vector, 0, 0);
        gsl_vector_set(&camera_q.vector, 1, 0);
        gsl_vector_set(&camera_q.vector, 2, 0);
        gsl_vector_set(&camera_q.vector, 3, 1);
        gsl_vector_set_zero(&camera_p.vector);
    }

    gsl_matrix_set_identity(msckf->P);

    return msckf;
}

void msckf_destroy(msckf_t *msckf) {
    gsl_vector_free(msckf->x);
    gsl_matrix_free(msckf->P);
    free(msckf);
}

void msckf_propagate(msckf_t *msckf, const double gyroscope[3], const double acceleration[3]) {
    gsl_vector_view imu = gsl_vector_subvector(msckf->x, 0, 16);
    gsl_vector_view q = gsl_vector_subvector(&imu.vector, 0, 4);
    gsl_vector_view bg = gsl_vector_subvector(&imu.vector, 4, 3);
    gsl_vector_view ba = gsl_vector_subvector(&imu.vector, 10, 3);
    gsl_vector_const_view wm = gsl_vector_const_view_array(gyroscope, 3);
    gsl_vector_const_view am = gsl_vector_const_view_array(acceleration, 3);

    double w_data[3];
    gsl_vector_view w = gsl_vector_view_array(w_data, 3);
    gsl_vector_memcpy(&w.vector, &wm.vector);
    gsl_vector_sub(&w.vector, &bg.vector);

    double a_data[3];
    gsl_vector_view a = gsl_vector_view_array(a_data, 3);
    gsl_vector_memcpy(&a.vector, &am.vector);
    gsl_vector_sub(&a.vector, &ba.vector);

    gsl_vector_view imu_params[2] = {w, a};
    double *state_imu_ptr = gsl_vector_ptr(&imu.vector, 0);
    intagrate(system_propagate_imu, state_imu_ptr, 16, IMU_T, imu_params);

    double c_data[3 * 3];
    gsl_matrix_view c = gsl_matrix_view_array(c_data, 3, 3);
    rotation_matrix(&q.vector, &c.matrix);

    double f_data[15 * 15];
    double a_cross_data[3 * 3];
    gsl_matrix_view f = gsl_matrix_view_array(f_data, 15, 15);
    gsl_matrix_view f11 = gsl_matrix_submatrix(&f.matrix, 0, 0, 3, 3);
    gsl_matrix_view f12 = gsl_matrix_submatrix(&f.matrix, 0, 3, 3, 3);
    gsl_matrix_view f31 = gsl_matrix_submatrix(&f.matrix, 6, 0, 3, 3);
    gsl_matrix_view a_cross = gsl_matrix_view_array(a_cross_data, 3, 3);
    gsl_matrix_view f34 = gsl_matrix_submatrix(&f.matrix, 6, 9, 3, 3);
    gsl_matrix_view f53 = gsl_matrix_submatrix(&f.matrix, 12, 6, 3, 3);
    gsl_matrix_set_zero(&f.matrix);
    cross_matrix(&w.vector, &f11.matrix);
    gsl_matrix_scale(&f11.matrix, -1);
    gsl_matrix_set_identity(&f12.matrix);
    gsl_matrix_scale(&f12.matrix, -1);
    cross_matrix(&a.vector, &a_cross.matrix);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1, &c.matrix, &a_cross.matrix, 0, &f31.matrix);
    gsl_matrix_transpose_memcpy(&f34.matrix, &c.matrix);
    gsl_matrix_scale(&f34.matrix, -1);
    gsl_matrix_set_identity(&f53.matrix);

    double g_data[15 * 12];
    gsl_matrix_view g = gsl_matrix_view_array(g_data, 15, 12);
    gsl_matrix_view g11 = gsl_matrix_submatrix(&f.matrix, 0, 0, 3, 3);
    gsl_matrix_view g22 = gsl_matrix_submatrix(&f.matrix, 3, 3, 3, 3);
    gsl_matrix_view g33 = gsl_matrix_submatrix(&f.matrix, 6, 6, 3, 3);
    gsl_matrix_view g44 = gsl_matrix_submatrix(&f.matrix, 9, 9, 3, 3);
    gsl_matrix_set_zero(&g.matrix);
    gsl_matrix_set_identity(&g11.matrix);
    gsl_matrix_scale(&g11.matrix, -1);
    gsl_matrix_set_identity(&g22.matrix);
    gsl_matrix_transpose_memcpy(&g33.matrix, &c.matrix);
    gsl_matrix_scale(&g33.matrix, -1);
    gsl_matrix_set_identity(&g44.matrix);

    gsl_matrix_view pii_params[2] = {f, g};
    gsl_matrix_view pii = gsl_matrix_submatrix(msckf->P, 0, 0, 15, 15);
    double *state_pii_ptr = gsl_matrix_ptr(&pii.matrix, 0, 0);
    intagrate(system_propagate_pii, state_pii_ptr, 15 * 15, IMU_T, pii_params);

    double phi_data[15 * 15];
    gsl_matrix_view phi = gsl_matrix_view_array(phi_data, 15, 15);
    gsl_matrix_set_identity(&phi.matrix);
    double *state_phi_ptr = gsl_matrix_ptr(&phi.matrix, 0, 0);
    intagrate(system_propagate_phi, state_phi_ptr, 15 * 15, IMU_T, &f);

    gsl_matrix *tmp = gsl_matrix_alloc(15, 6 * N);
    gsl_matrix_view pic = gsl_matrix_submatrix(msckf->P, 0, 15, 15, 6 * N);
    gsl_matrix_view pci = gsl_matrix_submatrix(msckf->P, 15, 0, 6 * N, 15);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &phi.matrix, &pic.matrix, 0, tmp);
    gsl_matrix_memcpy(&pic.matrix, tmp);
    gsl_matrix_transpose_memcpy(&pci.matrix, tmp);
    gsl_matrix_free(tmp);
}

void msckf_update(msckf_t *msckf,
                  const unsigned int id[],
                  const double x[],
                  const double y[],
                  const unsigned int num) {
    feature_t *f = msckf->features;
    while(f) {
        f->seen = 0;
        f = f->next;
    }

    for(size_t i = 0; i < num; i++) {
        const unsigned int fid = id[i];
        feature_t *f = feature_find(msckf->features, fid);
        if(!f) {
            f = feature_add(&msckf->features, fid);
        }
        feature_add_frame(f, x[i], y[i]);
        f->seen = 1;
    }

    feature_t **indirect = &msckf->features;
    while(*indirect) {
        feature_t *f = *indirect;
        if(!f->seen) {
            *indirect = f->next;
            free(f);
        } else {
            indirect = &f->next;
        }
    }

    for(size_t i = 0; i < N - 2; i++) {
        gsl_vector_view v = gsl_vector_subvector(msckf->x, 16 + (7 * (i + 0)), 7);
        gsl_vector_view w = gsl_vector_subvector(msckf->x, 16 + (7 * (i + 1)), 7);
        gsl_vector_swap(&v.vector, &w.vector);
    }

    const double qci_data[4] = {
        CAM_IMU_QX,
        CAM_IMU_QY,
        CAM_IMU_QZ,
        CAM_IMU_QW,
    };
    gsl_vector_const_view qci = gsl_vector_const_view_array(qci_data, 4);

    const double pci_data[3] = {
        CAM_IMU_PX,
        CAM_IMU_PY,
        CAM_IMU_PZ,
    };
    gsl_vector_const_view pci = gsl_vector_const_view_array(pci_data, 3);

    gsl_vector_view imu = gsl_vector_subvector(msckf->x, 0, 16);
    gsl_vector_view qig = gsl_vector_subvector(&imu.vector, 0, 4);
    gsl_vector_view pig = gsl_vector_subvector(&imu.vector, 4, 3);

    gsl_vector_view cam = gsl_vector_subvector(msckf->x, 16 + (7 * (N - 1)), 7);
    gsl_vector_view qcg = gsl_vector_subvector(&cam.vector, 0, 4);
    gsl_vector_view pcg = gsl_vector_subvector(&cam.vector, 4, 3);
    quaternion_product(&qci.vector, &qig.vector, &qcg.vector);

    double cig_data[3 * 3];
    gsl_matrix_view cig = gsl_matrix_view_array(cig_data, 3, 3);
    rotation_matrix(&qig.vector, &cig.matrix);

    gsl_vector_memcpy(&pcg.vector, &pig.vector);
    gsl_blas_dgemv(CblasTrans, 1, &cig.matrix, &pci.vector, 1, &pcg.vector);

    double cigtpic_data[3];
    gsl_matrix *ij = gsl_matrix_alloc(15 + (6 * N), 15 + (6 * N));
    gsl_matrix_view i = gsl_matrix_submatrix(ij, 0, 0, 15 + (6 * (N - 1)), 15 + (6 * N));
    gsl_matrix_view j = gsl_matrix_submatrix(ij, 15 + (6 * (N - 1)), 0, 6, 15 + (6 * N));
    gsl_matrix_view i_imu = gsl_matrix_submatrix(&i.matrix, 0, 0, 15, 15);
    gsl_matrix_view i_rest = gsl_matrix_submatrix(&i.matrix, 15, 15 + 6, 6 * (N - 1), 6 * (N - 1));
    gsl_matrix_view j11 = gsl_matrix_submatrix(&j.matrix, 0, 0, 3, 3);
    gsl_matrix_view j21 = gsl_matrix_submatrix(&j.matrix, 3, 0, 3, 3);
    gsl_vector_view cigtpic = gsl_vector_view_array(cigtpic_data, 3);
    gsl_matrix_view j23 = gsl_matrix_submatrix(&j.matrix, 3, 6, 3, 3);
    gsl_matrix_set_zero(&i.matrix);
    gsl_matrix_set_identity(&i_imu.matrix);
    gsl_matrix_set_identity(&i_rest.matrix);
    gsl_matrix_set_zero(&j.matrix);
    rotation_matrix(&qci.vector, &j11.matrix);
    gsl_blas_dgemv(CblasTrans, 1, &cig.matrix, &pci.vector, 0, &cigtpic.vector);
    cross_matrix(&cigtpic.vector, &j21.matrix);
    gsl_matrix_set_identity(&j23.matrix);

    gsl_matrix *tmp = gsl_matrix_alloc(15 + (6 * N), 15 + (6 * N));
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, ij, msckf->P, 0, tmp);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, tmp, ij, 0, msckf->P);
    gsl_matrix_free(tmp);

    gsl_matrix_free(ij);
}

void msckf_get(msckf_t *msckf, double quaternion[4], double position[3]) {
    gsl_vector_const_view imu_q = gsl_vector_const_subvector(msckf->x, 0, 4);
    gsl_vector_const_view imu_p = gsl_vector_const_subvector(msckf->x, 13, 3);

    quaternion[0] = gsl_vector_get(&imu_q.vector, 0);
    quaternion[1] = gsl_vector_get(&imu_q.vector, 1);
    quaternion[2] = gsl_vector_get(&imu_q.vector, 2);
    quaternion[3] = gsl_vector_get(&imu_q.vector, 3);

    position[0] = gsl_vector_get(&imu_p.vector, 0);
    position[1] = gsl_vector_get(&imu_p.vector, 1);
    position[2] = gsl_vector_get(&imu_p.vector, 2);
}
