#ifndef _IO_H_
#define _IO_H_

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "GKlib.h"

void writeFVec(std::vector<float> fvec, std::string fName);
void dispSpVec(gk_csr_t *mat, int row);
bool isFileExist(const char *fileName);

#endif
