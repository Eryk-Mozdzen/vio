#ifndef MSCKF_H
#define MSCKF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct msckf msckf_t;

msckf_t *msckf_create();
void msckf_destroy(msckf_t *msckf);

void msckf_propagate(msckf_t *msckf, const double gyroscope[3], const double acceleration[3]);
void msckf_update(msckf_t *msckf,
                  const unsigned int id[],
                  const double x[],
                  const double y[],
                  const unsigned int num);

void msckf_get(msckf_t *msckf, double quaternion[4], double position[3]);

#ifdef __cplusplus
}
#endif

#endif
