#include "io.h"



void writeFVec(std::vector<float> fvec, std::string fName) {
  std::ofstream opFile(fName);

  if (opFile.is_open()) {
    for (int i = 0; i < fvec.size(); i++) {
      opFile << fvec[i] << std::endl;
    }
    opFile.close();
  }

}


