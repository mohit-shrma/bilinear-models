#include "modelFactRMSE.h"

float ModelFactRMSE::objective(const Data& data) {
  
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

  norm = V.norm();
  vReg = norm*norm*l2Reg;

  obj += rmse + uReg + vReg;
  return obj;
}


void ModelFactRMSE::computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& uFeat, float r_ui) {
  float r_ui_est = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat);
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*V);
  Ugrad *= 2.0*(r_ui_est - r_ui);
  //regularization
  Ugrad += 2.0*l2Reg*U;
}


void ModelFactRMSE::computeVGrad(Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& uFeat, float r_ui) {
  float r_ui_est = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat);
  Vgrad = iFeat*((uFeat-iFeat).transpose()*U);
  Vgrad *= 2.0*(r_ui_est - r_ui);
  //regularization
  Vgrad += 2.0*l2Reg*V;
}


void gradCheck(Eigen::MatrixXf& Ugrad, Eigen::MatrixXf& Vgrad, 
    Eigen::VectorXf& iFeat, Eigen::VectorXf& uFeat, Eigen::MatrixXf& U,
    Eigen::MatrixXf& V, float r_ui) {
  
  float lossRight, lossLeft, gradE, lr_ll_e;
  float epsilon = 0.0001;
  float r_ui_est = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat);

  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*V);
  Ugrad *= 2.0*(r_ui_est - r_ui);
  
  Vgrad = iFeat*((uFeat-iFeat).transpose()*U);
  Vgrad *= 2.0*(r_ui_est - r_ui);
  
  //perturbation matrix
  Eigen::MatrixXf perturbMat = Eigen::MatrixXf::Zero(U.rows(), U.cols());
  int i = std::rand() % U.rows();
  int j = std::rand() % U.cols();
  perturbMat(i, j) = epsilon;

  //perturb U with +E and compute loss
  Eigen::MatrixXf noisyU = U + perturbMat;
  float r_ui_est2 = ((uFeat - iFeat).transpose()*noisyU)*(V.transpose()*iFeat);
  lossRight = (r_ui_est2 - r_ui)*(r_ui_est2 - r_ui);

  //perturb U with -E and compute loss
  noisyU = U - perturbMat;
  float r_ui_est3 = ((uFeat - iFeat).transpose()*noisyU)*(V.transpose()*iFeat);
  lossLeft = (r_ui_est3 - r_ui)*(r_ui_est3 - r_ui);
  
  gradE = Ugrad(i,j);
  lr_ll_e = (lossRight - lossLeft)/(2.0*epsilon); 
  if (fabs(lr_ll_e - gradE) > 0.001) {
    std::cout << "\nU lr: " << lossRight << " ll: " << lossLeft 
      << " gradE: " << gradE << " lr_ll_e: " << lr_ll_e 
      << " lr_ll_e-gradE: " << fabs(lr_ll_e - gradE) << " "
      << r_ui_est << " " << r_ui_est2 << " " << r_ui_est3 << std::endl; 
  }

  //perturb V with +E and compute loss
  Eigen::MatrixXf noisyV = V + perturbMat;
  r_ui_est2 = ((uFeat - iFeat).transpose()*U)*(noisyV.transpose()*iFeat);
  lossRight = (r_ui_est2 - r_ui)*(r_ui_est2 - r_ui);

  //perturb V with -E and compute loss
  noisyV = V - perturbMat;
  r_ui_est3 = ((uFeat - iFeat).transpose()*U)*(noisyV.transpose()*iFeat);
  lossLeft = (r_ui_est3 - r_ui)*(r_ui_est3 - r_ui);
  
  gradE = Vgrad(i,j);
  lr_ll_e = (lossRight - lossLeft)/(2.0*epsilon); 
  if (fabs(lr_ll_e - gradE) > 0.001) {
    std::cout << "\nV lr: " << lossRight << " ll: " << lossLeft 
      << " gradE: " << gradE << " lr_ll_e: " << lr_ll_e 
      << " lr_ll_e-gradE: " << fabs(lr_ll_e - gradE) << " "
      << r_ui_est << " " << r_ui_est2 << " " << r_ui_est3 << std::endl; 
  }

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

      //perform gad check
      //gradCheck(Ugrad, Vgrad, iFeat, uFeat, U, V, r_ui);

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
        << " best iter: " << bestIter << " best obj: " << bestObj;
      std::cout << "\nTrain RMSE: " << computeRMSE(data.trainMat, data) << std::endl;
    }

  }


}


