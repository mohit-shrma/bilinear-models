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
    void computeUSpGrad(int u, int i, int j, float r_uij, const Data& data, 
        Eigen::MatrixXf& Ugrad, Eigen::VectorXf& f_iT_V, Eigen::VectorXf& f_jT_V,
        Eigen::VectorXf& f_u_f_i_diff);
    void computeVGrad(int u, int i, int j, const Data& data, 
      Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
      Eigen::VectorXf& uFeat);
    void computeVSpGrad(int u, int i, int j, float r_uij, const Data& data, 
      Eigen::MatrixXf& Vgrad, Eigen::VectorXf& f_uT_U, Eigen::VectorXf& f_uT_U_f_iT_U);
    void gradCheck(int u, int i, int j, Eigen::MatrixXf& Ugrad, Eigen::MatrixXf& Vgrad,
      const Data& data, Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& jFeat);
};

#endif

