#ifndef _MODEL_FACT_SYM_BPR_H_
#define _MODEL_FACT_SYM_BPR_H_

#include "modelFactSym.h"

class ModelFactSymBPR: public ModelFactSym {

  public:
    ModelFactSymBPR(const Params &params, int nFeatures):ModelFactSym(params, nFeatures){}
    virtual void train(const Data& data, Model& bestModel);
    void computeUGrad(int u, int i, int j, const Data& data, 
      Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
      Eigen::VectorXf& uFeat);
    void computeUSpGrad(int u, int i, int j, const Data& data, 
      Eigen::MatrixXf& Ugrad, float r_uij_est, Eigen::VectorXf& f_iT_U,
      Eigen::VectorXf& f_uT_U, Eigen::VectorXf& f_jT_U, 
      Eigen::VectorXf& f_iT_U_f_jT_U, Eigen::VectorXf& f_uT_U_2f_iT_U) ;
};


#endif
