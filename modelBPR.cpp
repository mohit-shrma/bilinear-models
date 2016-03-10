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
  float lossRight, lossLeft, gradE;

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
  //w_norm = noisyW1.norm(); 
  lossRight = -log(sigmoid(r_ui - r_uj));// + l2Reg*w_norm*w_norm;

  //perturb W with -E and compute loss
  auto noisyW2 = W - perturbMat; 
  r_ui = (uFeat - iFeat).transpose()*noisyW2*iFeat;
  r_uj = uFeat.transpose()*noisyW2*jFeat;
  //w_norm = noisyW2.norm();
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

  int bestIter, u, pI, nI;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::MatrixXf T(nFeatures, nFeatures);
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float bestRecall, prevRecall, r_ui, r_uj;
  int trainNNZ = getNNZ(data.trainMat); 
 
  std::chrono::time_point<std::chrono::system_clock> start, end;
  
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
 
  
  start = std::chrono::system_clock::now();
  std::cout << "val recall: " << computeRecallParVec(data.valMat, data, 10, data.valItems) << std::endl;
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "\nValidation recall duration: " << duration.count() << std::endl;
  
  //random engine
  std::mt19937 mt(seed);
  
  auto uiRatings = getBPRUIRatings(data.trainMat);
  std::cout << "\nuiRatings size: " << uiRatings.size();

  double regMult = (1.0 - 2.0*learnRate*l2Reg);
  for (int iter = 0; iter < maxIter; iter++) {
    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    start = std::chrono::system_clock::now();
    T.fill(0);
    int subIter = 0;
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u    = std::get<0>(uiRating);
      pI   = std::get<1>(uiRating);
      
      //sample a negative item for user u
      nI = data.sampleNegItem(u);
      
      r_ui = estPosRating(u, pI, data, pdt);
      r_uj = estNegRating(u, nI, data, pdt);
      double r_uij = r_ui - r_uj;
      double expCoeff = 1.0 /(1.0 + exp(r_uij));
      
      //learnRate * expCoeff * f_u * f_i^T
      lazySparseUpdMatWSpOuterPdt(W, T, data.uFAccumMat, u, data.itemFeatMat, pI, 
          learnRate*expCoeff, regMult, subIter, l1Reg);
      
      //- learnRate * expCoeff * f_u * f_j^T
      lazySparseUpdMatWSpOuterPdt(W, T, data.uFAccumMat, u, data.itemFeatMat, nI, 
          -learnRate*expCoeff, regMult, subIter, l1Reg);

      //-learnRate * expCoeff * f_i * f_i^T
      lazySparseUpdMatWSpOuterPdt(W, T, data.itemFeatMat, pI, data.itemFeatMat, pI, 
          -learnRate*expCoeff, regMult, subIter, l1Reg);
      
      subIter++;
    } 
    
    //perform reg updates on all the pairs
    for (int ind1 = 0; ind1 < nFeatures; ind1++) {
      for (int ind2 = 0; ind2 < nFeatures; ind2++) {
        //update with reg updates
         W(ind1, ind2) = W(ind1, ind2)*pow(regMult, 
                                           subIter-T(ind1, ind2));
        //L1 or proximal update
        W(ind1, ind2) = proxL1(W(ind1, ind2), l1Reg);
      }
    }


    end = std::chrono::system_clock::now();
    duration = end - start;
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << "\niter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " duration: " 
        << duration.count() << std::endl;
      std::cout << "\nW norm: " << W.norm();
    }
  
  }
  
}

