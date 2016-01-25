#ifndef _MODEL_FACT_SYM_H_
#define _MODEL_FACT_SYM_H_

#include "model.h"

class ModelFactSym: public Model {
  
  public:
    Eigen::MatrixXf U;
    ModelFactSym(const Params &params, int nFeatures);
    
    virtual float estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt);
    virtual float estNegRating(int u, int item, const Data& data, 
      Eigen::VectorXf& pdt);
};

#endif

