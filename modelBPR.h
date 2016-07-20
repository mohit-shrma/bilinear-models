#ifndef _MODEL_BPR_H_
#define _MODEL_BPR_H_

#include "model.h"
#include "mathUtil.h"
#include "modelFullMat.h"
#include "util.h"
#include <array>
#include <cmath>
#include <unordered_set>

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

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
    void gradCheck(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad);
    void updUIRatings(std::vector<std::tuple<int, int, int>>& bprTriplets, 
      const Data& data, Eigen::MatrixXf& T, int& subIter, int nTrainSamp, int start, int end);
    void parTrain(const Data &data, Model& bestModel);
    void FTRLTrain(const Data &data, Model& bestModel);
    void ParFTRLTrain(const Data &data, Model& bestModel);
    void FTRLMiniGrad(std::vector<std::tuple<int, int, int>>& bprTriplets, 
        const Data& data, Eigen::MatrixXf& Wgrad, Eigen::MatrixXi& T, std::vector<bool>& done, 
        int thInd, int start, int end);
    void FTRLGradComp(Eigen::MatrixXf& Wgrad, MatrixXb& T, 
        Eigen::MatrixXf& z, Eigen::MatrixXf& n, gk_csr_t* mat1, int row1, 
        gk_csr_t *mat2, int row2);
    void FTRLGradUpd(Eigen::MatrixXf& Wgrad, MatrixXb& T, 
        Eigen::MatrixXf& z, Eigen::MatrixXf& n, gk_csr_t* mat1, int row1, 
        gk_csr_t *mat2, int row2);
    void computeBPRSparseGradWOReset(int u, int i, int j, 
        Eigen::MatrixXf& Wgrad, Eigen::MatrixXi& T, Eigen::VectorXf& pdt, const Data& data);
};


#endif

