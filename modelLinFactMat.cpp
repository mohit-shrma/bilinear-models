#include "modelLinFactMat.h"


ModelLinFactMat::ModelLinFactMat(const Params& params, int nFeatures)
  :Model(params, nFeatures) {
  //initialize factored matrices
  U = Eigen::MatrixXf(nFeatures, rank);
  V = Eigen::MatrixXf(nFeatures, rank);
  w = Eigen::VectorXf(nFeatures);
  for (int i = 0; i < nFeatures; i++) {
    w(i) = 1.0;//(float)std::rand()/ (float)(1.0 + RAND_MAX); 
    for (int j = 0; j < rank; j++) {
        U(i, j) = (float)std::rand()/ (float)(1.0 + RAND_MAX);
        V(i, j) = (float)std::rand()/ (float)(1.0 + RAND_MAX);
    }
  }
  std::cout << "\nw norm: " << w.norm();
  std::cout << "\nU dim: " << U.rows() << " " << U.cols() << " U norm: " << U.norm();
  std::cout << "\nV dim: " << V.rows() << " " << V.cols() << " V norm: " << V.norm();
}


float ModelLinFactMat::estPosRating(int u, int item, const Data& data, 
    Eigen::VectorXf& pdt) {
  float r_ui = 0;
  //TODO: removed this temporary creation
  Eigen::VectorXf pdt1(rank);
  Eigen::VectorXf pdt2(rank);
  pdt1.fill(0);
  pdt2.fill(0);
  
  //(f_u^TU - f_i^TU)
  spVecMatPdt(U, data.uFAccumMat, u, pdt1); 
  spVecMatPdt(U, data.itemFeatMat, item, pdt2);
  pdt1 = pdt1 - pdt2;

  //f_i^TV
  spVecMatPdt(V, data.itemFeatMat, item, pdt2);
  r_ui = pdt1.dot(pdt2);
  
  //f_u .* w .*f_i
  r_ui += spVecWtspVecPdt(w, data.itemFeatMat, item, data.uFAccumMat, u);
  //-f_i .* w .* f_i
  r_ui -= spVecWtspVecPdt(w, data.itemFeatMat, item, data.itemFeatMat, item);

  return r_ui;
}


float ModelLinFactMat::estNegRating(int u, int item, const Data& data, 
    Eigen::VectorXf& pdt) {
  float r_ui = 0;
  //TODO: removed this temporary creation
  Eigen::VectorXf pdt1(rank);
  Eigen::VectorXf pdt2(rank);
  pdt1.fill(0);
  pdt2.fill(0);
  
  //f_u^TU
  spVecMatPdt(U, data.uFAccumMat, u, pdt1); 

  //f_i^TV
  spVecMatPdt(V, data.itemFeatMat, item, pdt2);
  
  r_ui = pdt1.dot(pdt2);

  //f_u .* w .*f_i
  r_ui += spVecWtspVecPdt(w, data.itemFeatMat, item, data.uFAccumMat, u);
  
  return r_ui;
}








