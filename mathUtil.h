#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

#include <Eigen/Dense>
#include <iostream>
#include "GKlib.h"
extern "C" {
  #include "svdlib.h"
}

float vecSpVecDot(Eigen::VectorXf& vec, gk_csr_t *mat, int row);
void performNucNormProj(Eigen::MatrixXf& W, float gamma);
void performNucNormProjSVDLib(Eigen::MatrixXf& W, int rank);
void performNucNormProjSVDLibWReg(Eigen::MatrixXf& W, float gamma);
float sigmoid(float x);
void matSpVecPdt(Eigen::MatrixXf& W, gk_csr_t *mat, int row, 
    Eigen::VectorXf& pdt);
void spVecMatPdt(Eigen::MatrixXf& W, gk_csr_t *mat, int row, 
    Eigen::VectorXf& pdt);
void updateMatWSpOuterPdt(Eigen::MatrixXf& W, gk_csr_t *mat1, int row1, 
    gk_csr_t *mat2, int row2, float scalar);
void updateMatWSymSpOuterPdt(Eigen::MatrixXf& W, gk_csr_t *mat1, int row1, 
    gk_csr_t *mat2, int row2, float scalar);
Eigen::MatrixXf spMatMatPdt(gk_csr_t *mat, Eigen::MatrixXf& W);
void spVecDiff(gk_csr_t* mat1, int row1, gk_csr_t* mat2, int row2, 
    Eigen::VectorXf& res);
void spVecVecOuterPdt(Eigen::MatrixXf& pdt, Eigen::VectorXf& vec, gk_csr_t* mat,
    int row);
float sparseDotProd2(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j);
float sparseDotProd(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j);
float spVecWtspVecPdt(Eigen::VectorXf& w, gk_csr_t* mat1, int row1, 
    gk_csr_t* mat2, int row2);
void lazyUpdMatWSpOuterPdt(Eigen::MatrixXf& W, Eigen::MatrixXf& T, 
    gk_csr_t *mat1, int row1, gk_csr_t *mat2, int row2, double scalar, 
    double regMult, int subIter);
#endif
