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

void ModelBPR::train(const Data &data, Model& bestModel) {
  
  

}


