#include "modelFactRMSE.h"


void computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& uFeat, float r_ui) {
  float r_ui_est = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat);
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*V);
  Ugrad *= 2.0*(r_ui_est - r_ui);
}


void computeVGrad(Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& uFeat, float r_ui) {
  float r_ui_est = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat);
  Vgrad = iFeat*((uFeat-iFeat).transpose()*U);
  Vgrad *= 2.0*(r_ui_est - r_ui);
}


void ModelFactRMSE::train(const Data &data, Model& bestModel) {
  
  std::cout << "\nModelFactRMSE::train";

  int bestIter;
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::MatrixXf Vgrad(nFeatures, rank);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  float bestObj, prevObj, valRecall;
  int trainNNZ = getNNZ(data.trainMat); 

  int u, ii, item, nUserItems;
  float r_ui;
   
  std::cout <<"\nB4 Train Objective: " << objective(data) << std::endl;

  for (int iter = 0; iter < maxIter; iter++) {
    std::chrono::time_point<std::chrono::system_clock> startSub, endSub;
    startSub = std::chrono::system_clock::now();
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

      //compute U gradient
      computeUGrad(Ugrad, iFeat, uFeat, r_ui);
      //update U
      U -= learnRate*Ugrad;
      
      //compute U gradient
      computeVGrad(Vgrad, iFeat, uFeat, r_ui);
      //update V
      V -= learnRate*Vgrad;

    }
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      //valRecall = computeRecallPar(data.valMat, data, 10, data.valItems);
      if (isTerminateModelObj(bestModel, data, iter, bestIter, bestObj, 
          prevObj)) {
        break;
      }
      std::cout << "\nIter: " << iter << " obj: " << prevObj
        << " best iter: " << bestIter << " best obj: " << bestObj << std::endl;
      std::cout << "\nTrain RMSE: " << computeRMSE(data.trainMat, data) << std::endl;
    }

  }


}


