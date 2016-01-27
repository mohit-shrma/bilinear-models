#include "modelLinear.h"

ModelLinear::ModelLinear(const Params &params, 
    int nFeatures):Model(params, nFeatures) {
  w = Eigen::VectorXf(nFeatures); 
  for (int i = 0; i < nFeatures; i++) {
    w(i) = (float)std::rand()/ (float)(1.0 + RAND_MAX); 
  }
  std::cout << "\n w norm: " << w.norm();
}


float ModelLinear::estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt) {
  float r_ui = 0;
  //(f_u-f_i)*w*f_i
  Eigen::VectorXf fu_fi(nFeatures);
  spVecDiff(data.uFAccumMat, u, data.itemFeatMat, item, fu_fi); 
  
  fu_fi = fu_fi.*w;
}


float ModelLinear::estNegRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt) {
  
}
