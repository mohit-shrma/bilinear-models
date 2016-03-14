#include <iostream>
#include <chrono>
#include "datastruct.h"
#include "model.h"
#include "modelBPR.h"
#include "modelRMSE.h"

#include "modelFactBPR.h"
#include "modelFactRMSE.h"
#include "modelCosine.h"
#include "modelFactSymBPR.h"
#include "modelFactSymRMSE.h"
#include "modelLinearRMSE.h"
#include "modelLinearBPR.h"
#include "modelLinFactMatBPR.h"
#include "modelRMSEFullGrad.h"
#include "modelBPRFullGrad.h"

Params parse_cmd_line(int argc, char *argv[]) {
  
  if (argc < 16) {
    std::cout  << "\nNot enough arguments";
    exit(0);
  } 

  Params params(argv[1], argv[2], argv[3], argv[4], argv[5],
      atof(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]),
      atoi(argv[11]), atoi(argv[12]), atof(argv[13]), atoi(argv[14]), 
      atoi(argv[15]));
  
  return params;
}


int main(int argc, char *argv[]) {

  
  //get passed parameters
  Params params = parse_cmd_line(argc, argv);

  //initialize random generator
  std::srand(params.seed);

  Data data(params);
  
  std::chrono::time_point<std::chrono::system_clock> start, end;
 
  //create baseline model
  //ModelCosine cosModel(params, data.nFeatures);

  /*
  start = std::chrono::system_clock::now();
  float baseRecallPar = cosModel.computeRecallPar(data.testMat, data, 10, 
      data.testItems);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "\nTest baseline recall par: " << baseRecallPar << " " 
    << duration.count() << std::endl;
  */

  //float baseValRecallPar = cosModel.computeRecallPar(data.valMat, data, 10, 
  //    data.valItems);
  //std::cout << "\nVal baseline recall par: " << baseValRecallPar << std::endl;
  
  //ModelFactMat bestModel(params, data.nFeatures);
  ModelFullMat bestModel(params, data.nFeatures);
 
  //ModelRMSEFGrad m(params, data.nFeatures);
  //ModelBPRFGrad m(params, data.nFeatures);
  //ModelBPR m(params, data.nFeatures);
  //ModelRMSE m(params, data.nFeatures);  
  ModelBPR m(params, data.nFeatures);

  m.train(data, bestModel);
  
  float testRecall = bestModel.computeRecallParVec(data.testMat, data, 10, 
      data.testItems);
  std::cout << "\nTest recall: " << testRecall;

  float valRecall = bestModel.computeRecallParVec(data.valMat, data, 10, 
      data.valItems);
  std::cout << "\nVal recall: " << valRecall;

  std::cout << "\nRE: " << params.l2Reg << " " << params.l1Reg << " "
    << params.wReg << " " << params.nucReg << " " << params.learnRate << " " 
    << params.rank << " " << valRecall << " " << testRecall << std::endl;
   
  return 0;
}



