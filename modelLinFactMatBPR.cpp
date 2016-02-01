#include "modelLinFactMatBPR.h"


void ModelLinFactMatBPR::computeUGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) {
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat); 
  r_ui  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  r_uj  += uFeat.transpose()*(jFeat.cwiseProduct(w));
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*V) - uFeat*(jFeat.transpose()*V);
  Ugrad *= expCoeff;
  //regularization
  Ugrad += 2.0*l2Reg*U;
}


void ModelLinFactMatBPR::computeVGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) {
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat); 
  r_ui  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  r_uj  += uFeat.transpose()*(jFeat.cwiseProduct(w));
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  Vgrad = (iFeat - jFeat)*(uFeat.transpose()*U) - iFeat*(iFeat.transpose()*U);
  Vgrad *= expCoeff;
  //regularization
  Vgrad += 2.0*l2Reg*V;
}


void ModelLinFactMatBPR::computewGrad(int u, int i, int j, const Data& data, 
    Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat, 
    Eigen::VectorXf& uFeat) {
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat); 
  r_ui  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  r_uj  += uFeat.transpose()*(jFeat.cwiseProduct(w));
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  wgrad = ((uFeat - iFeat).cwiseProduct(iFeat)) - uFeat.cwiseProduct(jFeat);
  wgrad *= expCoeff;
  wgrad += 2.0*wReg*w;
}


void ModelLinFactMatBPR::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelLinFactMatBPR::train" << std::endl;

  int bestIter, u, i, j;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::MatrixXf Vgrad(nFeatures, rank);  

  float bestRecall, prevRecall;
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
      
      extractFeat(data.uFAccumMat, u, uFeat);
      extractFeat(data.itemFeatMat, i, iFeat);
      extractFeat(data.itemFeatMat, j, jFeat);

      //compute w gradient
      computewGrad(u, i, j, data, wgrad, iFeat, jFeat, uFeat);
      w -= learnRate*wgrad;
      
      //compute U gradient
      computeUGrad(u, i, j, data, Ugrad, iFeat, jFeat, uFeat);
      U -= learnRate*Ugrad;

      //compute V gradient
      computeVGrad(u, i, j, data, Vgrad, iFeat, jFeat, uFeat);
      V -= learnRate*Vgrad;
    } 
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << std::endl <<"iter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " w norm: " << w.norm()
        << " U norm: " << U.norm() << " V norm: " << V.norm();
    }
  
  }
  
}



