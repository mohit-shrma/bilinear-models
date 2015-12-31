#include <iostream>
#include "datastruct.h"
#include "model.h"


Params parse_cmd_line(int argc, char *argv[]) {
  
  if (argc < 13) {
    std::cout  << "\nNot enough arguments";
    exit(0);
  }

  Params params(argv[1], argv[2], argv[3], argv[4],
      atof(argv[5]), atof(argv[6]), atof(argv[7]), atoi(argv[8]),
      atoi(argv[9]), atof(argv[10]), atoi(argv[11]), atoi(argv[12]));
  
  return params;
}


int main(int argc, char *argv[]) {

  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  //initialize random generator
  std::srand(params.seed);

  Data data(params);
  
  //create baseline model
  Model cosineModel(params, data.nFeatures);
  cosineModel.W = MatrixXd::Identity(data.nFeatures, data.nFeatures);
  float baseRecall = cosineModel.computeRecall(data.testMat, data, 10, 
      data.testItems);
  
  //create bpr model
  //TODO:

  return 0;
}



