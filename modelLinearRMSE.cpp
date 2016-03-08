#include "modelLinearRMSE.h"


float ModelLinearRMSE::objective(const Data& data) {
  
  int u, ii, item, nnz;
  float r_ui, r_ui_est, norm;
  Eigen::VectorXf pdt(rank);
  float rmse = 0, w_reg = 0, obj = 0;
  nnz = 0;
  for (u = 0; u < data.trainMat->nrows; u++) {
    for (ii = data.trainMat->rowptr[u]; 
        ii < data.trainMat->rowptr[u+1]; ii++) {
      item = data.trainMat->rowind[ii];
      r_ui_est = estPosRating(u, item, data, pdt);
      r_ui = data.trainMat->rowval[ii];
      rmse += (r_ui_est - r_ui)*(r_ui_est-r_ui);
      nnz++;
    }
  }
 
  norm = w.norm();
  w_reg = norm*norm*wReg;

  std::cout << "\nmse: " << rmse/nnz << " w_reg: " << w_reg << " w norm: " << w.norm(); 

  obj = rmse/nnz + w_reg;
  return obj;
}


void ModelLinearRMSE::train(const Data& data, Model& bestModel) {
  std::cout << "\nModelLinearRMSE::train";

  int bestIter;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);
  float bestObj, prevObj;
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
      
      //sample a rated item
      nUserItems = data.trainMat->rowptr[u+1] - data.trainMat->rowptr[u];
      ii = std::rand()%nUserItems + data.trainMat->rowptr[u];
      item = data.trainMat->rowind[ii];
      r_ui = data.trainMat->rowval[ii];

      extractFeat(data.uFAccumMat, u, uFeat);
      extractFeat(data.itemFeatMat, item, iFeat);

      //r_ui_est
      r_ui_est = ((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w));
      wgrad = 2.0*(r_ui_est - r_ui)*iFeat.cwiseProduct(uFeat - iFeat);
      wgrad += 2.0*wReg*w;

      //update U
      w -= learnRate*wgrad;
    }
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
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

