#ifndef _IO_H_
#define _IO_H_

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "GKlib.h"

void writeFVec(std::vector<float> fvec, std::string fName);
void dispSpVec(gk_csr_t *mat, int row);
bool isFileExist(const char *fileName);
std::vector<float> readFVector(const char *ipFileName);
void readEigenVec(const char* fileName, Eigen::VectorXf& vec, int size);
void readEigenMat(const char* fileName, Eigen::MatrixXf& mat, int nrows, 
    int ncols);
void readBinEigenMat(const char* fileName, Eigen::MatrixXf& mat, int nrows, 
    int ncols);
void writeEigenMat(Eigen::MatrixXf& mat, std::string& fName);
void writeBinEigenMat(Eigen::MatrixXf& mat, std::string& fName);
void writeEigenVec(Eigen::VectorXf& vec, std::string& fName);

#endif
