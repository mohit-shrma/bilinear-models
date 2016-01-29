#include "modelBPR.h"



void ModelBPR::computeBPRGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad) {
  double r_ui, r_uj, r_uij, expCoeff;
  
  r_ui  = (uFeat - iFeat).transpose()*W*iFeat;
  r_uj  = uFeat.transpose()*W*jFeat;
  r_uij = r_ui - r_uj;
   
  Wgrad.fill(0);
  expCoeff = -1.0/(1.0 + exp(r_uij));  
  //need to update W as j has higher preference
  Wgrad = ((uFeat - iFeat)*iFeat.transpose()
                    - uFeat*jFeat.transpose());
  Wgrad *= expCoeff;
  Wgrad += 2.0*l2Reg*W;
}


void ModelBPR::computeBPRSparseGrad(int u, int i, int j, 
    Eigen::MatrixXf& Wgrad, Eigen::VectorXf& pdt, const Data& data) {
  
  float r_ui, r_uj, r_uij, expCoeff;
   
  r_ui  = estPosRating(u, i, data, pdt);
  r_uj  = estNegRating(u, j, data, pdt);
  r_uij = r_ui - r_uj;
   
  Wgrad.fill(0);
  //need to update W as j has higher preference
  expCoeff = 1.0/(1.0 + exp(r_uij));  

  //-f_u*f_i^T
  updateMatWSpOuterPdt(Wgrad, data.uFAccumMat, u, data.itemFeatMat, i, 
      -1);

  //f_u*f_j^T
  updateMatWSpOuterPdt(Wgrad, data.uFAccumMat, u, data.itemFeatMat, j, 
      1);

  //f_i*f_i^T
  updateMatWSpOuterPdt(Wgrad, data.itemFeatMat, i, data.itemFeatMat, i, 
      1);
  
  Wgrad *= expCoeff;
  
  //add Wgrad to gradient of l2 reg
  Wgrad += 2.0*l2Reg*W;

}


void ModelBPR::gradCheck(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad) {
  
  double r_ui, r_uj, r_uij, expCoeff;
  float w_norm, lossRight, lossLeft, gradE;

  r_ui  = (uFeat - iFeat).transpose()*W*iFeat;
  r_uj  = uFeat.transpose()*W*jFeat;
  r_uij = r_ui - r_uj;

  if (r_uij > 0 ) {
    return;
  }

  expCoeff = 1.0/(1.0 + exp(r_uij));  
   
  //reset Wgrad to gradient of l2 reg
  //Wgrad = 2.0*l2Reg*W;
  Wgrad = -expCoeff*((uFeat - iFeat)*iFeat.transpose()
                      - uFeat*jFeat.transpose());
  //perturbation matrix
  auto perturbMat = Eigen::MatrixXf::Constant(W.rows(), W.cols(), 0.0001);

  //perturb W with +E and compute loss
  auto noisyW1 = W + perturbMat; 
  r_ui = (uFeat - iFeat).transpose()*noisyW1*iFeat;
  r_uj = uFeat.transpose()*noisyW1*jFeat;
  w_norm = noisyW1.norm(); 
  lossRight = -log(sigmoid(r_ui - r_uj));// + l2Reg*w_norm*w_norm;

  //perturb W with -E and compute loss
  auto noisyW2 = W - perturbMat; 
  r_ui = (uFeat - iFeat).transpose()*noisyW2*iFeat;
  r_uj = uFeat.transpose()*noisyW2*jFeat;
  w_norm = noisyW2.norm();
  lossLeft = -log(sigmoid(r_ui - r_uj));// + l2Reg*w_norm*w_norm;

  //compute gradient and E dotprod
  gradE = 2.0*(Wgrad.cwiseProduct(perturbMat).sum());

  if (fabs(lossRight - lossLeft - gradE) > 0.001) {
    std::cout << "\nlr: " << lossRight << " ll: " << lossLeft 
      << " gradE: " << gradE  
      << " lr-ll: " << lossRight-lossLeft 
      << "\n(lr-ll)/gradE: " << (lossRight-lossLeft)/gradE  
      << " lr-ll-gradE: " << lossRight-lossLeft-gradE
      << " uFeatNorm: " << uFeat.norm()
      << " iFeatNorm: " << iFeat.norm() << " jFeatNorm: " << jFeat.norm()
      << std::endl;
  }
  
}


void ModelBPR::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelBPR::train" << std::endl;

  int bestIter;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float bestRecall, prevRecall;
  int trainNNZ = getNNZ(data.trainMat); 
  std::array<int, 3> triplet;
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
  std::cout << "val recall: " << computeRecallPar(data.valMat, data, 10, data.valItems) << std::endl;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  for (int iter = 0; iter < maxIter; iter++) {
    start = std::chrono::system_clock::now();
    for (int subIter = 0; subIter < trainNNZ*pcSamples; subIter++) {
        
      //sample triplet
      triplet = data.sampleTriplet();
      uFeat = data.uFeatAcuum.row(triplet[0]); 
      extractFeat(data.itemFeatMat, triplet[1], iFeat);
      extractFeat(data.itemFeatMat, triplet[2], jFeat);
      
      //compute gradient
      //gradCheck(uFeat, iFeat, jFeat, Wgrad); 
      //computeBPRGrad(uFeat, iFeat, jFeat, Wgrad);
      computeBPRSparseGrad(triplet[0], triplet[1], triplet[2], Wgrad, pdt, 
          data);

      //update W
      W -= learnRate*Wgrad;
    } 
    
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    //TODO:nuclear norm projection on each triplet or after all sub-iters
    performNucNormProjSVDLib(W, rank);
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << "\niter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " duration: " 
        << duration.count() << std::endl;
    }
  
  }
  
}


