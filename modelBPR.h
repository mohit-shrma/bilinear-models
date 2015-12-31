#ifndef _MODEL_BPR_H_
#define _MODEL_BPR_H_

#include "model.h"
#include "mathUtil.h"
#include <array>

class ModelBPR: public Model {
  
  public:
    ModelBPR(const Params &params, int nFeatures):Model(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    std::array<int, 3> sampleTriplet(const Data &data); 
    void computeBPRGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat,
        Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad);
};


#endif

