#ifndef _MODEL_BPR_H_
#define _MODEL_BPR_H_

#include "model.h"
#include "mathUtil.h"
#include "modelFullMat.h"
#include "util.h"
#include <array>
#include <cmath>

class ModelBPR: public ModelFullMat {
  
  public:
    ModelBPR(const Params &params, int nFeatures):ModelFullMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    virtual bool isTerminateModel(Model& bestModel, const Data& data, int iter,
      int& bestIter, float& bestRecall, float& prevRecall);
    void computeBPRGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat,
      Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad);
    void computeBPRSparseGrad(int u, int i, int j, 
      Eigen::MatrixXf& Wgrad, Eigen::VectorXf& pdt, const Data& data);
    void gradCheck(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad);
};


#endif

