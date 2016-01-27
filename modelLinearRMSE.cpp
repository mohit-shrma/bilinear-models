#include "modelLinearRMSE.h"


float ModelLinearRMSE::objective(const Data& data) {
  
  int u, ii, item;
  float r_ui, r_ui_est, norm;
  Eigen::VectorXf pdt(rank);
  float rmse = 0, wReg = 0, obj = 0;

  for (u = 0; u < data.trainMat->nrows; u++) {
    for (ii = data.trainMat->rowptr[u]; 
        ii < data.trainMat->rowptr[u+1]; ii++) {
      item = data.trainMat->rowind[ii];
      r_ui_est = estPosRating(u, item, data, pdt);
      r_ui = data.trainMat->rowval[ii];
      rmse += (r_ui_est - r_ui)*(r_ui_est-r_ui);
    }
  }
 
  norm = w.norm();
  wReg = norm*norm*l2Reg;

  std::cout << "\nse: " << rmse << " wReg: " << wReg << " w norm: " << w.norm(); 

  obj = rmse + wReg;
  return obj;
}


void ModelLinearRMSE::train(const Data& data, Model& bestModel) {
  std::cout << "\nModelLinearRMSE::train";

  int bestIter;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);
  float bestObj, prevObj, valRecall;
  int trainNNZ = getNNZ(data.trainMat); 

  int u, ii, item, nUserItems;
  float r_ui, r_ui_est;
   
  std::cout <<"\nB4 Train Objective: " << objective(data) << std::endl;
  std::cout << "\nTrain RMSE: " << computeRMSE(data.trainMat, data) << std::endl;

  for (int iter = 0; iter < maxIter; iter++) {
    std::chrono::time_point<std::chrono::system_clock> startSub, endSub;
    startSub = std::chrono::system_clock::now();
    for (int subIter = 0; subIter < trainNNZ; subIter++) {
    //for (int subIter = 0; subIter < 10; subIter++) {
      
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

      //r_ui_est
      r_ui_est = ((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w));
      wgrad = 2.0*(r_ui_est - r_ui)*iFeat.cwiseProduct(uFeat - iFeat);
      wgrad += 2.0*l2Reg*w;

      //update U
      w -= learnRate*wgrad;
    }
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      //valRecall = computeRecallPar(data.valMat, data, 10, data.valItems);
      if (isTerminateModelObj(bestModel, data, iter, bestIter, bestObj, 
                              prevObj)) {
        break;
      }
      std::cout << "\nIter: " << iter << " obj: " << prevObj
        << " best iter: " << bestIter << " best obj: " << bestObj 
        << " Train RMSE: " << computeRMSE(data.trainMat, data);
    }

  }



}

