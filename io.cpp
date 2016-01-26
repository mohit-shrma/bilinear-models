#include "io.h"



void writeFVec(std::vector<float> fvec, std::string fName) {
  std::ofstream opFile(fName);

  if (opFile.is_open()) {
    for (int i = 0; i < fvec.size(); i++) {
      opFile << i << " " << fvec[i] << std::endl;
    }
    opFile.close();
  }

}


void dispSpVec(gk_csr_t *mat, int row) {
  std::cout << std::endl;
  for (int ii = mat->rowptr[row]; ii < mat->rowptr[row+1]; ii++) {
    std::cout << mat->rowind[ii] << "," << mat->rowval[ii] << " ";
  }
}

