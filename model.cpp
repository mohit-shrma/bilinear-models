#include "model.h"


Model::Model(const Params &params, int p_nFeatures) {
  nFeatures = p_nFeatures;
  l2Reg     = params.l2Reg;
  nucReg    = params.nucReg;
  learnRate = params.learnRate;
  rank      = params.rank;
  maxIter   = params.maxIter;
  
  //initialize model matrix
  W = Eigen::MatrixXf::Zero(nFeatures, nFeatures);
  for (int i = 0; i < nFeatures; i++) {
    for (int j = 0; j < nFeatures; j++) {
      W(i,j) = (float)std::rand()/ (float)(1.0 + RAND_MAX);
    }
  }
  

}


