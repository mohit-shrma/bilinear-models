#include "modelLinSymRMSE.h"


float ModelLinSymRMSE::objective(const Data& data) {

  int u, ii, item;
  float r_ui, r_ui_est, norm;
  Eigen::VectorXf pdt(rank);
  float rmse = 0, uReg = 0, w_reg = 0, obj = 0;

  for (u = 0; u < data.trainMat->nrows; u++) {
    for (ii = data.trainMat->rowptr[u]; 
        ii < data.trainMat->rowptr[u+1]; ii++) {
      item = data.trainMat->rowind[ii];
      r_ui_est = estPosRating(u, item, data, pdt);
      r_ui = data.trainMat->rowval[ii];
      rmse += (r_ui_est - r_ui)*(r_ui_est-r_ui);
    }
  }
 
  norm = U.norm();
  uReg = norm*norm*l2Reg;
  
  norm = w.norm();
  w_reg = norm*norm*wReg;

  std::cout << "\nse: " << rmse << " uReg: " << uReg << " U norm: " << U.norm()
    << " w_reg: " << w_reg << " w norm: " << w.norm(); 

  obj = rmse + uReg + w_reg;
  return obj;
}


void ModelLinSymRMSE::computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& uFeat, float r_ui) {
  float r_ui_est = ((uFeat - iFeat).transpose()*U)*(U.transpose()*iFeat);
  r_ui_est  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*U) + iFeat*((uFeat - iFeat).transpose()*U);
  Ugrad *= 2.0*(r_ui_est - r_ui);
  //regularization
  Ugrad += 2.0*l2Reg*U;
}


void ModelLinSymRMSE::computewGrad(Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& uFeat, float r_ui) {
  float r_ui_est = ((uFeat - iFeat).transpose()*U)*(U.transpose()*iFeat);
  r_ui_est  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  wgrad = ((uFeat - iFeat).cwiseProduct(iFeat));
  wgrad *= 2.0*(r_ui_est - r_ui);
  wgrad += 2.0*wReg*w;
}


void ModelLinSymRMSE::train(const Data &data, Model& bestModel) {
  
  std::cout << "\nModelLinSymRMSE::train";

  int bestIter;
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);
  float bestObj, prevObj, valRecall;
  int trainNNZ = getNNZ(data.trainMat); 

  int u, ii, item, nUserItems;
  float r_ui, r_ui_est;
   
  std::cout <<"\nB4 Train Objective: " << objective(data);
  std::cout << "\nTrain RMSE: " << computeRMSE(data.trainMat, data);

  for (int iter = 0; iter < maxIter; iter++) {
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


      //compute U gradient
      computeUGrad(Ugrad, iFeat, uFeat, r_ui);

      //update U
      U -= learnRate*Ugrad;
      
      //compute w grad
      computewGrad(wgrad, iFeat, uFeat, r_ui);

      //update w
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

