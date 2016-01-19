#ifndef _MODEL_FACT_BPR_H_
#define _MODEL_FACT_BPR_H_

#include "modelFactMat.h"

class ModelFactBPR: public ModelFactMat {

  public:
    ModelFactBPR(const Params &params, int nFeatures):ModelFactMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    void computeUGrad(int u, int i, int j, const Data& data, 
      Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
      Eigen::VectorXf& uFeat);
    void computeVGrad(int u, int i, int j, const Data& data, 
      Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
      Eigen::VectorXf& uFeat);
};

#endif

