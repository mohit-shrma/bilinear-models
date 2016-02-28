#include "modelFactSymRMSE.h"

float ModelFactSymRMSE::objective(const Data& data) {

  int u, ii, item;
  float r_ui, r_ui_est, norm;
  Eigen::VectorXf pdt(rank);
  float rmse = 0, uReg = 0, vReg = 0, obj = 0;

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

  std::cout << "\nse: " << rmse << " uReg: " << uReg << " U norm: " << U.norm(); 

  obj = rmse + uReg + vReg;
  return obj;
}


void ModelFactSymRMSE::computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& uFeat, float r_ui) {
  float r_ui_est = ((uFeat - iFeat).transpose()*U)*(U.transpose()*iFeat);
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*U) + iFeat*((uFeat - iFeat).transpose()*U);
  Ugrad *= 2.0*(r_ui_est - r_ui);
  //regularization
  Ugrad += 2.0*l2Reg*U;
}


void ModelFactSymRMSE::computeUSpGrad(Eigen::MatrixXf& Ugrad, const Data& data,
    float r_ui, float r_ui_est, int u, int i, Eigen::VectorXf& f_iT_U, 
    Eigen::VectorXf& f_uT_U) {
  
  Ugrad.fill(0);
  //-2f_if_i^TU
  spVecVecOuterPdt(Ugrad, f_iT_U, data.itemFeatMat, i);
  Ugrad *= -2;

  //f_i*f_u^TU
  spVecVecOuterPdt(Ugrad, f_uT_U, data.itemFeatMat, i);
  //f_u*f_i^TU
  spVecVecOuterPdt(Ugrad, f_iT_U, data.uFAccumMat, u);

  Ugrad *= 2.0*(r_ui_est - r_ui);
  //regularization
  Ugrad += 2.0*l2Reg*U;
}


void ModelFactSymRMSE::train(const Data& data, Model& bestModel) {
  std::cout << "\nModelFactSymRMSE::train";

  int bestIter;
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::MatrixXf Vgrad(nFeatures, rank);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf f_uT_U(rank);
  Eigen::VectorXf f_iT_U(rank);
  Eigen::VectorXf f_u_f_i_diff(nFeatures);
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
      
      //sample a rated item
      nUserItems = data.trainMat->rowptr[u+1] - data.trainMat->rowptr[u];
      ii = std::rand()%nUserItems + data.trainMat->rowptr[u];
      item = data.trainMat->rowind[ii];
      r_ui = data.trainMat->rowval[ii];
     
      //compute f_u^TU
      spVecMatPdt(U, data.uFAccumMat, u, f_uT_U);
      
      //compute f_i^TU
      spVecMatPdt(U, data.itemFeatMat, item, f_iT_U);

      //compute f_u - f_i
      spVecDiff(data.uFAccumMat, u, data.itemFeatMat, item, f_u_f_i_diff);

      //r_ui_est
      r_ui_est = f_iT_U.dot(f_uT_U - f_iT_U);

      extractFeat(data.uFAccumMat, u, uFeat);
      extractFeat(data.itemFeatMat, item, iFeat);

      //perform grad check
      //gradCheck(Ugrad, Vgrad, iFeat, uFeat, U, V, r_ui);

      //compute U gradient
      computeUGrad(Ugrad, iFeat, uFeat, r_ui);
      //computeUSpGrad(Ugrad, data, r_ui, r_ui_est, u, item, f_iT_U, f_uT_U);

      //update U
      U -= learnRate*Ugrad;
      
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

