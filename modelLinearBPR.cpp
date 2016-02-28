#include "modelLinearBPR.h"


void ModelLinearBPR::computewGrad(int u, int i, int j, const Data& data, 
    Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat, 
    Eigen::VectorXf& uFeat) {
  float r_uij_est = (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w))); 
  r_uij_est -= uFeat.transpose()*(jFeat.cwiseProduct(w));
  float expCoeff = -1.0/(1.0 + exp(r_uij_est));
  wgrad = ((uFeat - iFeat).cwiseProduct(iFeat)) - uFeat.cwiseProduct(jFeat);
  wgrad *= expCoeff;
  wgrad += 2.0*wReg*w;
}


void ModelLinearBPR::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelLinearBPR::train" << std::endl;

  int bestIter, u, i, j;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);

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
    } 
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << std::endl <<"iter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " w norm: " << w.norm();
    }
  
  }
  
}


