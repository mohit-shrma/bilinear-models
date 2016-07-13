#include "io.h"



void writeFVec(std::vector<float> fvec, std::string fName) {
  std::ofstream opFile(fName);

  if (opFile.is_open()) {
    for (size_t i = 0; i < fvec.size(); i++) {
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


bool isFileExist(const char *fileName) {
  std::ifstream infile(fileName);
  return infile.good();
}


std::vector<float> readFVector(const char *ipFileName) {
  std::vector<float> vec;
  std::ifstream ipFile(ipFileName);
  std::string line; 
  if (ipFile.is_open()) {
    while(getline(ipFile, line)) {
      if (line.length() > 0) {
        vec.push_back(std::stof(line));
      }
    }
    ipFile.close();
  } else {
    std::cerr <<  "\nCan't open file: " << ipFileName;
    exit(0);
  }
  return vec;
}


void readEigenVec(const char* fileName, Eigen::VectorXf& vec, int size) {
  std::cout << "Reading... " << fileName << std::endl;
  auto fvec = readFVector(fileName);
  vec.setZero();
  if (fvec.size() != size) {
    std::cerr << "Size mismatch: " << fileName << " " << fvec.size() << " " 
      << size << std::endl;  
  } else {
    for (int i = 0; i < size; i++) {
      vec(i) = fvec[i];
    }  
  }
  std::cout << "vec norm: " << vec.norm() << std::endl;
}


//TODO:verify
void readEigenMat(const char* fileName, Eigen::MatrixXf& mat, int nrows, 
    int ncols) {
  
  size_t pos;
  std::string line, token;
  std::ifstream ipFile(fileName);
  std::string delimiter = " ";
  int rowInd = 0, colInd = 0;
  mat.setZero();
  if (ipFile.is_open()) {
    std::cout << "Reading... " << fileName << std::endl;  
    while (getline(ipFile, line)) {
      colInd = 0;
      while((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        mat(rowInd, colInd) = std::stof(token);
        colInd++;
        line.erase(0, pos + delimiter.length());
      }
      if (line.length() > 0) {
        mat(rowInd, colInd) = std::stof(line);
      }
      rowInd++;
    } 
    ipFile.close();
  }
  
  std::cout << "Read: nrows: " << rowInd << " ncols: " << colInd << std::endl;
  std::cout << "mat norm: " << mat.norm() << std::endl;
}


void writeEigenMat(Eigen::MatrixXf& mat, std::string& fName) {
  int nrows = mat.rows();
  int ncols = mat.cols();
  
  if (nrows > 0 && ncols > 0) {
    std::ofstream opFile(fName);
    if (opFile.is_open()) {
      std::cout << "Writing... " << fName << " " << nrows << " " 
        << ncols << std::endl;
      for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
          opFile << mat(i, j) << " ";
        }
        opFile << std::endl;
      }
      opFile.close();
    }
  }

}


void readBinEigenMat(const char* fName, Eigen::MatrixXf& mat, int numRows, 
    int numCols) {
  int nrows, ncols;
  std::ifstream inFile(fName, std::ios::in|std::ios::binary);
  if (inFile.is_open()) {
    std::cout << "Reading..." << fName << std::endl;
    inFile.read((char*) (&nrows), sizeof(int));
    inFile.read((char*) (&ncols), sizeof(int));
    std::cout << "nrows: " << nrows << " ncols: " << ncols << std::endl;
    
    if (numRows != nrows || numCols != ncols) {
      std::cerr << "rows n cols mismatch: " << nrows << " " << numRows << " " 
        << ncols << " " << numCols << std::endl;
    }

    inFile.read((char*) mat.data(), nrows*ncols*sizeof(typename Eigen::MatrixXf::Scalar));
    inFile.close();
  }
  
  std::cout << "mat norm: " << mat.norm() << std::endl;

}


void writeBinEigenMat(Eigen::MatrixXf& mat, std::string& fName) {
  int nrows = mat.rows();
  int ncols = mat.cols();
  std::ofstream opFile(fName, std::ios::out|std::ios::binary);
  if (opFile.is_open()) {
    std::cout << "Writing... " << fName << " " << nrows << " " 
      << ncols << std::endl;
    opFile.write((char*) (&nrows), sizeof(int));
    opFile.write((char*) (&ncols), sizeof(int));
    opFile.write((char*) mat.data(), nrows*ncols*sizeof(typename Eigen::MatrixXf::Scalar));
    opFile.close();
  }

}


void writeEigenVec(Eigen::VectorXf& vec, std::string& fName) {
  if (vec.size() > 0) {
    std::ofstream opFile(fName);
    if (opFile.is_open()) {
      std::cout << "Writing... " << fName << std::endl;
      for (int i = 0; i < vec.size(); i++) {
        opFile << vec(i) << std::endl;
      }
      opFile.close();
    }
  }
}


