#include "modelRMSE.h"


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
  float bestRecall, prevRecall;
  int trainNNZ = getNNZ(data.trainMat); 

  int u, ii, item, nUserItems;
  float r_ui;

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
      computeGrad(uFeat, iFeat, Wgrad, r_ui);

      //update W
      W -= learnRate*Wgrad;
    }
    
    //TODO:nuclear norm projection on each triplet or after all sub-iters
    //performNucNormProj(W, nucReg);
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall);
      std::cout << "\niter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall;
    }

  }


}

