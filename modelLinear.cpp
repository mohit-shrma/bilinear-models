#include "modelLinear.h"

ModelLinear::ModelLinear(const Params &params, 
    int nFeatures):Model(params, nFeatures) {
  w = Eigen::VectorXf(nFeatures); 
  for (int i = 0; i < nFeatures; i++) {
    w(i) = 1.0;//(float)std::rand()/ (float)(1.0 + RAND_MAX); 
  }
  std::cout << "\n w norm: " << w.norm();
}


float ModelLinear::estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt) {
  float r_ui = 0;
  //compute (f_u-f_i) .* w .* f_i
  //f_u .* w .*f_i
  r_ui = spVecWtspVecPdt(w, data.itemFeatMat, item, data.uFAccumMat, u);
  //-f_i .* w .* f_i
  r_ui -= spVecWtspVecPdt(w, data.itemFeatMat, item, data.itemFeatMat, item);
  return r_ui;
}


float ModelLinear::estNegRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt) {
  //compute f_u .* w .* f_i
  //f_u .* w .*f_i
  float r_ui = spVecWtspVecPdt(w, data.itemFeatMat, item, data.uFAccumMat, u);
  return r_ui;  
}


