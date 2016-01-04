#include "modelRMSE.h"


float ModelRMSE::objective(const Data& data) {
  
  int u, ii, item;
  float r_ui, r_ui_est, w_norm;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  float rmse = 0, uReg = 0, nucNormReg = 0, nucNorm = 0, obj = 0;

  for (u = 0; u < data.trainMat->nrows; u++) {
    uFeat = data.uFeatAcuum.row(u);
    for (ii = data.trainMat->rowptr[u]; 
        ii < data.trainMat->rowptr[u+1]; ii++) {
      item = data.trainMat->rowind[ii];
      extractFeat(data.itemFeatMat, item, iFeat);
      r_ui_est = (uFeat - iFeat).transpose()*W*iFeat;
      r_ui = data.trainMat->rowval[ii];
      rmse += (r_ui_est - r_ui)*(r_ui_est-r_ui);
    }
    w_norm = W.norm();
    uReg += l2Reg*w_norm*w_norm;
  }
  
  //compute thin svd
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(W, 
      Eigen::ComputeThinU|Eigen::ComputeThinV);
  auto singVec = svd.singularValues();
  for (ii = 0; ii < nFeatures; ii++) {
    nucNorm += singVec[ii]; 
  }
  nucNormReg = nucNorm*nucReg;
  
  obj += rmse + uReg + nucNormReg;
  
  return obj;
}


void ModelRMSE::computeGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, Eigen::MatrixXf& Wgrad, float r_ui) {
  float r_ui_est;
  
  r_ui_est  = (uFeat - iFeat).transpose()*W*iFeat;
  
  //std::cout << "\nr_ui: " << r_ui << " r_ui_est: " << r_ui_est << std::endl;

  //reset Wgrad to gradient of l2 reg
  Wgrad = 2.0*l2Reg*W;
  Wgrad += 2.0*(r_ui_est - r_ui)*((uFeat - iFeat)*iFeat.transpose());
}


//gradient check
void ModelRMSE::gradCheck(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
    Eigen::MatrixXf& Wgrad, float r_ui) {
  
  float r_ui_est, lossRight, lossLeft, gradE, w_norm;
  
  r_ui_est  = (uFeat - iFeat).transpose()*W*iFeat;
   
  //reset Wgrad to gradient of l2 reg
  Wgrad = 2.0*l2Reg*W;
  Wgrad += 2.0*(r_ui_est - r_ui)*((uFeat - iFeat)*iFeat.transpose());

  //perturbation matrix
  auto perturbMat = Eigen::MatrixXf::Constant(W.rows(), W.cols(), 0.0001);

  //perturb W with +E and compute loss
  auto noisyW1 = W + perturbMat; 
  r_ui_est  = (uFeat - iFeat).transpose()*noisyW1*iFeat;
  w_norm = noisyW1.norm();
  lossRight = (r_ui_est - r_ui)*(r_ui_est - r_ui) + l2Reg*w_norm*w_norm;

  //perturb W with -E and compute loss
  auto noisyW2 = W - perturbMat; 
  r_ui_est  = (uFeat - iFeat).transpose()*noisyW2*iFeat;
  w_norm = noisyW2.norm();
  lossLeft = (r_ui_est - r_ui)*(r_ui_est - r_ui) + l2Reg*w_norm*w_norm;

  //compute gradient and E dotprod
  gradE = 2.0*(Wgrad.cwiseProduct(perturbMat).sum());
  
  if (fabs(lossRight - lossLeft - gradE) > 0.01) {
    std::cout << "\nlr: " << lossRight << " ll: " << lossLeft 
      << " gradE: " << gradE  
      << " lr-ll: " << lossRight-lossLeft 
      << "\n(lr-ll)/gradE: " << (lossRight-lossLeft)/gradE  
      << " lr-ll-gradE: " << lossRight-lossLeft-gradE << std::endl;
  }

}


void ModelRMSE::train(const Data &data, Model& bestModel) {
  
  int bestIter;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  float bestObj, prevObj, valRecall;
  int trainNNZ = getNNZ(data.trainMat); 

  int u, ii, item, nUserItems;
  float r_ui;
  
  std::cout <<"\nB4 Train Objective: " << objective(data) << std::endl;

  for (int iter = 0; iter < maxIter; iter++) {
    for (int subIter = 0; subIter < trainNNZ; subIter++) {
      
      //sample user
      while (1) {
        u = std::rand() % data.nUsers;
        if (data.posTrainUsers.find(u) != data.posTrainUsers.end()) {
          //found u
          break;
        }
      }
      
      //sample a pos rated item
      nUserItems = data.trainMat->rowptr[u+1] - data.trainMat->rowptr[u];
      while (1) {
        ii = std::rand()%nUserItems + data.trainMat->rowptr[u];
        item = data.trainMat->rowind[ii];
        r_ui = data.trainMat->rowval[ii];
        if (r_ui > 0) {
          break;
        }
      }
      
      uFeat = data.uFeatAcuum.row(u);
      extractFeat(data.itemFeatMat, item, iFeat);

      //compute gradient
      //gradCheck(uFeat, iFeat, Wgrad, r_ui);
      computeGrad(uFeat, iFeat, Wgrad, r_ui);

      //update W
      W -= learnRate*Wgrad;      

      //std::cout << "\n" << W.block<3,3>(0,0) << std::endl;
      //std::cout << W << std::endl;
    }
      
    //nuclear norm projection after all sub-iters
    performNucNormProj(W, nucReg);
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      valRecall = computeRecall(data.valMat, data, 10, data.valItems);
      if (isTerminateModelObj(bestModel, data, iter, bestIter, bestObj, 
          prevObj)) {
        break;
      }
      std::cout << std::endl << "Iter: " << iter << " obj: " << prevObj
        << " best iter: " << bestIter << " best obj: " << bestObj 
        << " val recall: " << valRecall << " W norm: " << W.norm();
      std::cout << "\nTrain RMSE: " << computeRMSE(data.trainMat, data);
    }

  }


}

