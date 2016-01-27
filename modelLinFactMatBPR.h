#ifndef _MODEL_LIN_FACT_MAT_BPR_H_
#define _MODEL_LIN_FACT_MAT_BPR_H_

#include "modelLinFactMat.h"

class ModelLinFactMatBPR: public ModelLinFactMat {

  public:
    ModelLinFactMatBPR(const Params &params, int nFeatures):ModelLinFactMat(params, nFeatures){}

    virtual void train(const Data &data, Model& bestModel);
    void computeUGrad(int u, int i, int j, const Data& data, 
      Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
      Eigen::VectorXf& uFeat);
    void computeVGrad(int u, int i, int j, const Data& data, 
      Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
      Eigen::VectorXf& uFeat);
    void computewGrad(int u, int i, int j, const Data& data, 
      Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat, 
      Eigen::VectorXf& uFeat);

};

#endif
