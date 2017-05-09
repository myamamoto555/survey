#include "NMT.hpp"

#define print(var)  \
  std::cout<<(var)<<std::endl

int main(int argc, char** argv){
  const int inputDim = 5;
  const int hiddenEncDim = 5;
  const int hiddenDim = 5; 
  const Real scale = 2.1;
  const bool useBlackOut = true;
  const int blackOutSampleNum = 3;
  const Real blackOutAlpha = 0.4;
  const Real clipThreshold = 3.0;
  const int beamSize = 20;
  const int maxLen = 100;
  const int miniBatchSize = 1;
  const int threadNum = 1;
  const Real learningRate = 1.1;
  const bool learningRateSchedule = false;
  const int srcVocaThreshold = 2;
  const int tgtVocaThreshold = 2;
  std::ostringstream saveDirName;
  std::ostringstream loadModelName;
  saveDirName << "./result/";
  Eigen::initParallel();

  /* Training Data */
  const std::string srcTrain = "data/train.en.1"; // 変更
  const std::string tgtTrain = "data/train.ja.1"; // 変更

  /* Development Data */
  /*
  const std::string srcDev = "data/dev.en.1"; // 変更
  const std::string tgtDev = "data/dev.ja.1"; // 変更
  */
  /* gradient checking*/
  const std::string srcDev = "data/train.en.1";
  const std::string tgtDev = "data/train.ja.1";

  
  // NMT
  NMT::demo(srcTrain, tgtTrain,
	    srcDev, tgtDev, 
	    inputDim, hiddenEncDim, hiddenDim,
	    scale, useBlackOut, blackOutSampleNum, blackOutAlpha,
	    clipThreshold,
	    beamSize, maxLen, miniBatchSize, threadNum,
	    learningRate, learningRateSchedule,
	    srcVocaThreshold, tgtVocaThreshold,
	    saveDirName.str());
  
  return 0;
}
