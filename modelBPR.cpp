#include "modelBPR.h"

std::array<int, 3> ModelBPR::sampleTriplet(const Data &data) {
  
  std::array<int, 3> triplet{{-1, -1, -1}};
  int u = -1, i = -1, j = -1, ii, jj;
  int nUserItems, start, end;
  int nTrainItems = data.trainItems.size();
  int32_t *ui_rowind = data.trainMat->rowind;
  ssize_t *ui_rowptr = data.trainMat->rowptr;
  float   *ui_rowval = data.trainMat->rowval;
  
  //sample user
  while (1) {
    u = std::rand() % data.nUsers;
    if (data.posTrainUsers.find(u) != data.posTrainUsers.end()) {
      //found u
      break;
    }
  }
  triplet[0] = u;

  //sample pos item
  nUserItems = ui_rowptr[u+1] - ui_rowptr[u];
  while (1) {
    ii = std::rand()%nUserItems + ui_rowptr[u];
    i = ui_rowind[ii];
    if (ui_rowval[ii] > 0) {
      break;
    }
  }
  triplet[1] = i;

  //sample neg item
  while(1) {
    jj = std::rand()%nUserItems;
    if (ui_rowval[jj + ui_rowptr[u]] == 0.0) {
      //explicit 0
      j = ui_rowind[jj + ui_rowptr[u]];
      break;
    } else {
      //search for implicit 0
      
      //TODO: do == check once all the code is finished
      if (0 == jj) {
        start = 0;
        end = ui_rowind[ui_rowptr[u]]; //first rated item by u
      } else if (nUserItems-1 == jj) {
        start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to last rated item
        end = nTrainItems;
      } else {
        start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to jjth item
        end = ui_rowind[ui_rowptr[u] + jj + 1]; //item rated after jjth item
      }

      //check for empty interval
      if (end - start > 0) {
        j = std::rand()%(end-start) + start;
      } else {
        continue;
      }

      //make sure sampled -ve item not present in testSet and valSet
      if (data.testItems.find(j) != data.testItems.end() ||
          data.valItems.find(j) != data.valItems.end()) {
        //found in either set
        continue;
      }

      if (data.trainItems.find(j) != data.trainItems.end()) {
        break;
      }
    }
  } //end while

  triplet[2] = j;

  return triplet;
}


void ModelBPR::computeBPRGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad) {
  double r_ui, r_uj, r_uij, expCoeff;
  
  r_ui  = (uFeat - iFeat).transpose()*W*iFeat;
  r_uj  = uFeat.transpose()*W*jFeat;
  r_uij = r_ui - r_uj;
  expCoeff = 1.0/(1.0 + exp(r_uij));  
   
  //reset Wgrad to gradient of l2 reg
  Wgrad = 2.0*l2Reg*W;
  Wgrad -= expCoeff*((uFeat - iFeat)*iFeat.transpose()
                      - uFeat*jFeat.transpose());
}



void ModelBPR::train(const Data &data, Model& bestModel) {

  int u, i, j;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);

  int trainNNZ = getNNZ(data.trainMat); 
  
  std::array<int, 3> triplet;
  
  for (int iter = 0; iter < maxIter; iter++) {
    for (int subIter = 0; subIter < trainNNZ; subIter++) {
        
      //sample triplet
      triplet = sampleTriplet(data);
      uFeat = data.uFeatAcuum.row(triplet[0]); 
      extractFeat(data.itemFeatMat, triplet[1], iFeat);
      extractFeat(data.itemFeatMat, triplet[2], jFeat);
      
      //compute gradient
      computeBPRGrad(uFeat, iFeat, jFeat, Wgrad);

      //update W
      W -= learnRate*Wgrad;

      //TODO:nuclear norm projection on each triplet or after all sub-iters
      performNucNormProj(W, nucReg);
    }
    //perform model evaluation on validation set
  }

}


