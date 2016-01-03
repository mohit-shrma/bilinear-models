#include "modelRMSE.h"

//TODO:
void gradCheck() {
}


float ModelRMSE::objective(const Data& data) {
  
  int u, ii, item;
  float r_ui, r_ui_est;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  float rmse = 0, uReg = 0, nucNormReg = 0, nucNorm = 0, obj = 0;

  for (u = 0; u < data.trainMat->nrows; u++) {
    uFeat = data.uFeatAcuum.row(u);
    for (ii = data.trainMat->rowptr[u]; 
        ii < data.trainMat->rowptr[u+1]; ii++) {
      extractFeat(data.itemFeatMat, item, iFeat);
      item = data.trainMat->rowind[ii];
      r_ui_est = (uFeat - iFeat).transpose()*W*iFeat;
      r_ui = data.trainMat->rowval[ii];
      rmse += (r_ui_est - r_ui)*(r_ui_est-r_ui);
    }
    uReg += l2Reg*W.norm();
  }

  //compute thin svd
  /*
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(W, 
      Eigen::ComputeThinU|Eigen::ComputeThinV);
  auto singVec = svd.singularValues();
  for (ii = 0; ii < nFeatures; ii++) {
    nucNorm += singVec[ii]; 
  }
  nucNormReg = nucNorm*nucReg;
  */

  obj += rmse + uReg + nucNormReg;
  
  return obj;
}


void ModelRMSE::computeGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, Eigen::MatrixXf& Wgrad, float r_ui) {
  float r_ui_est;
  
  r_ui_est  = (uFeat - iFeat).transpose()*W*iFeat;
   
  //reset Wgrad to gradient of l2 reg
  Wgrad = 2.0*l2Reg*W;
  Wgrad += 2.0*(r_ui_est - r_ui)*((uFeat - iFeat)*iFeat.transpose());
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
    for (int subIter = 0; subIter < 10; subIter++) {
      
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
      computeGrad(uFeat, iFeat, Wgrad, r_ui);

      //update W
      W -= learnRate*Wgrad;
      //TODO:nuclear norm projection on each triplet or after all sub-iters
      performNucNormProj(W, nucReg);
      
      std::cout << "\nsubIter: " << subIter << " W norm: " << 
        W.norm() << std::endl;

      //std::cout << W << std::endl;
    }
    
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      valRecall = computeRecall(data.valMat, data, 10, data.valItems);
      isTerminateModelObj(bestModel, data, iter, bestIter, bestObj, 
          prevObj);
      std::cout << "\nIter: " << iter << " prev obj: " << prevObj
        << " best obj: " << bestObj << " val recall: " << valRecall 
        << std::endl;
    }

  }


}

