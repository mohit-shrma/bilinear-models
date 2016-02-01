#include "modelFactBPR.h"

void ModelFactBPR::computeUGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) {
 
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat);
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  float r_uij = r_ui - r_uj;
  
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*V) - uFeat*(jFeat.transpose()*V);
  Ugrad *= expCoeff;
  //regularization
  Ugrad += 2.0*l2Reg*U;
}


void ModelFactBPR::computeUSpGrad(int u, int i, int j, float r_uij, const Data& data, 
    Eigen::MatrixXf& Ugrad, Eigen::VectorXf& f_iT_V, Eigen::VectorXf& f_jT_V,
    Eigen::VectorXf& f_u_f_i_diff) {
 
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  Ugrad.fill(0);
  spVecVecOuterPdt(Ugrad, f_jT_V, data.uFAccumMat, u);
  Ugrad = f_u_f_i_diff*(f_iT_V.transpose()) - Ugrad;
  Ugrad *= expCoeff;
  //regularization
  Ugrad += 2.0*l2Reg*U;
}


void ModelFactBPR::computeVGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) {
 
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat);
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  float r_uij = r_ui - r_uj;
  
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  Vgrad = (iFeat - jFeat)*(uFeat.transpose()*U) - iFeat*(iFeat.transpose()*U);
  Vgrad *= expCoeff;
  //regularization
  Vgrad += 2.0*l2Reg*V;
}


void ModelFactBPR::computeVSpGrad(int u, int i, int j, float r_uij, const Data& data, 
    Eigen::MatrixXf& Vgrad, Eigen::VectorXf& f_uT_U, Eigen::VectorXf& f_uT_U_f_iT_U) {
 
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  //f_i(f_u^TU - f_i^TU)
  Vgrad.fill(0);
  spVecVecOuterPdt(Vgrad, f_uT_U, data.itemFeatMat, j);
  Vgrad *= -1;
  spVecVecOuterPdt(Vgrad, f_uT_U_f_iT_U, data.itemFeatMat, i);
  Vgrad *= expCoeff;
  //regularization
  Vgrad += 2.0*l2Reg*V;
}


void ModelFactBPR::gradCheck(int u, int i, int j, Eigen::MatrixXf& Ugrad, Eigen::MatrixXf& Vgrad,
    const Data& data, Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& jFeat) {

  float lossRight, lossLeft, gradE, lr_ll_e, delta;
  float epsilon = 0.0001;
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat);
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));

  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*V) - uFeat*(jFeat.transpose()*V);
  Ugrad *= expCoeff;

  Vgrad = (iFeat - jFeat)*(uFeat.transpose()*U) - iFeat*(iFeat.transpose()*U);
  Vgrad *= expCoeff;

  //perturbation matrix
  Eigen::MatrixXf perturbMat = Eigen::MatrixXf::Zero(U.rows(), U.cols());
  int i1 = std::rand() % U.rows();
  int j1 = std::rand() % U.cols();
  perturbMat(i1, j1) = epsilon;

  //perturb U with +E and compute loss
  Eigen::MatrixXf noisyU = U + perturbMat;
  r_ui = ((uFeat - iFeat).transpose()*noisyU)*(V.transpose()*iFeat);
  r_uj = (uFeat.transpose()*noisyU)*(V.transpose()*jFeat);
  r_uij = r_ui - r_uj;
  lossRight = -log(sigmoid(r_uij));

  //perturb U with -E and compute loss
  noisyU = U - perturbMat;
  r_ui = ((uFeat - iFeat).transpose()*noisyU)*(V.transpose()*iFeat);
  r_uj = (uFeat.transpose()*noisyU)*(V.transpose()*jFeat);
  r_uij = r_ui - r_uj;
  lossLeft = -log(sigmoid(r_uij));

  gradE = Ugrad(i1, j1);
  lr_ll_e = (lossRight - lossLeft)/(2.0*epsilon); 
  delta = lr_ll_e - gradE;
  if (fabs(delta) > 0.001) {
    std::cout << "\nU lr: " << lossRight << " ll: " << lossLeft 
      << " gradE: " << gradE << " lr_ll_e: " << lr_ll_e 
      << " lr_ll_e-gradE: " << fabs(delta) << std::endl;
  }


  //perturb V with +E and compute loss
  Eigen::MatrixXf noisyV = V + perturbMat;
  r_ui = ((uFeat - iFeat).transpose()*U)*(noisyV.transpose()*iFeat);
  r_uj = (uFeat.transpose()*U)*(noisyV.transpose()*jFeat);
  r_uij = r_ui - r_uj;
  lossRight = -log(sigmoid(r_uij));
   
  //perturb V with -E and compute loss
  noisyV = V - perturbMat;
  r_ui = ((uFeat - iFeat).transpose()*U)*(noisyV.transpose()*iFeat);
  r_uj = (uFeat.transpose()*U)*(noisyV.transpose()*jFeat);
  r_uij = r_ui - r_uj;
  lossLeft = -log(sigmoid(r_uij));

  gradE = Vgrad(i1 ,j1);
  lr_ll_e = (lossRight - lossLeft)/(2.0*epsilon); 
  delta = lr_ll_e - gradE;
  if (fabs(delta) > 0.001) {
    std::cout << "\nV lr: " << lossRight << " ll: " << lossLeft 
      << " gradE: " << gradE << " lr_ll_e: " << lr_ll_e 
      << " lr_ll_e-gradE: " << fabs(delta) << std::endl;
  }

}


void ModelFactBPR::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelFactBPR::train" << std::endl;

  int bestIter, u, i, j;
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::MatrixXf Vgrad(nFeatures, rank);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf f_uT_U(rank);
  Eigen::VectorXf f_iT_U(rank);
  Eigen::VectorXf f_iT_V(rank);
  Eigen::VectorXf f_jT_V(rank);
  Eigen::VectorXf f_u_f_i_diff(nFeatures);

  float bestRecall, prevRecall, r_uij_est;
  int trainNNZ = getNNZ(data.trainMat); 
  std::array<int, 3> triplet;
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
  //std::cout << "val recall: " << computeRecallPar(data.valMat, data, 10, data.valItems) << std::endl;
  for (int iter = 0; iter < maxIter; iter++) {
    for (int subIter = 0; subIter < trainNNZ*pcSamples; subIter++) {
    //for (int subIter = 0; subIter < 10000; subIter++) {
        
      //sample triplet
      triplet = data.sampleTriplet();
      u = triplet[0];
      i = triplet[1];
      j = triplet[2];
      
      //extractFeat(data.uFAccumMat, u, uFeat);
      //extractFeat(data.itemFeatMat, i, iFeat);
      //extractFeat(data.itemFeatMat, j, jFeat);

      //gradCheck(u, i, j, Ugrad, Vgrad, data, uFeat, iFeat, jFeat);

      //compute f_u^TU
      spVecMatPdt(U, data.uFAccumMat, u, f_uT_U);
      
      //compute f_i^TU
      spVecMatPdt(U, data.itemFeatMat, i, f_iT_U);

      //compute f_i^TV
      spVecMatPdt(V, data.itemFeatMat, i, f_iT_V);
      
      //compute f_j^TV
      spVecMatPdt(V, data.itemFeatMat, j, f_jT_V);
      
      //compute f_u - f_i
      spVecDiff(data.uFAccumMat, u, data.itemFeatMat, i, f_u_f_i_diff);

      //r_uij_est
      r_uij_est = f_iT_V.dot(f_uT_U - f_iT_U) - f_jT_V.dot(f_uT_U);

      //compute U gradient
      computeUGrad(u, i, j, data, Ugrad, iFeat, jFeat, uFeat);
      //computeUSpGrad(u, i, j, r_uij_est, data, Ugrad, f_iT_V, f_jT_V, 
      //    f_u_f_i_diff);
      //update U
      U -= learnRate*Ugrad;

      //compute V gradient
      f_jT_V = f_uT_U - f_iT_U;
      computeVGrad(u, i, j, data, Vgrad, iFeat, jFeat, uFeat);
      //computeVSpGrad(u, i, j, r_uij_est, data, Vgrad, f_uT_U, f_jT_V);
      //update V
      V -= learnRate*Vgrad;
    } 
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << std::endl <<"iter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " U norm: " << U.norm()
        << " V norm: " << V.norm();
    }
  
  }
      std::cout << std::endl << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " U norm: " << U.norm()
        << " V norm: " << V.norm();
  
}

