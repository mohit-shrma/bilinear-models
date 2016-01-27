#ifndef _MODEL_FACT_MAT_H_
#define _MODEL_FACT_MAT_H_


#include "model.h"

class ModelFactMat: public Model {
  
  public:
    ModelFactMat(const Params &params, int nFeatures);
    
    virtual float estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt);
    virtual float estNegRating(int u, int item, const Data& data, 
      Eigen::VectorXf& pdt);

};


#endif

