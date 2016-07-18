#ifndef _MODEL_BPR_H_
#define _MODEL_BPR_H_

#include "model.h"
#include "mathUtil.h"
#include "modelFullMat.h"
#include "util.h"
#include <array>
#include <cmath>
#include <unordered_set>

class ModelBPR: public ModelFullMat {
  
  public:
    ModelBPR(const Params &params, int nFeatures):ModelFullMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    bool isTerminateModelInit(Model& bestModel, const Data& data, int iter,
      int& bestIter, float& bestRecall, float& prevRecall);
    void computeBPRGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat,
      Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad);
    void computeBPRSparseGrad(int u, int i, int j, 
      Eigen::MatrixXf& Wgrad, Eigen::VectorXf& pdt, const Data& data);
    void computeBPRSparseGrad(int u, int i, int j, 
    Eigen::MatrixXf& Wgrad, Eigen::VectorXf& pdt, const Data& data,
    std::map<int, std::unordered_set<int>>& coords);
    void gradCheck(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad);
    void updUIRatings(std::vector<std::tuple<int, int, int>>& bprTriplets, 
      const Data& data, Eigen::MatrixXf& T, int& subIter, int nTrainSamp, int start, int end);
    void parTrain(const Data &data, Model& bestModel);
    void FTRLTrain(const Data &data, Model& bestModel);
};


#endif

