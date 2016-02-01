#ifndef _MODEL_LIN_SYM_H_
#define _MODEL_LIN_SYM_H_

#include "model.h"

class ModelLinSym: public Model {
  
  public:
    ModelLinSym(const Params &params, int nFeatures);
    virtual float estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt);
    virtual float estNegRating(int u, int item, const Data& data, 
      Eigen::VectorXf& pdt);
};


#endif
