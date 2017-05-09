#pragma once

#include "LSTM.hpp"
#include "TreeLSTM.hpp"
#include "Vocabulary.hpp"
#include "SoftMax.hpp"
#include "BlackOut.hpp"
#include "Affine.hpp"

class NMT{
public:

  class Data;
  class Grad;
  class DecCandidate;
  class ThreadArg;

  enum OPT{
    SGD,
    MOMENTUM,
  };

  NMT::OPT opt;

  NMT(Vocabulary& sourceVoc_,
      Vocabulary& targetVoc_,
      std::vector<NMT::Data*>& trainData_,
      std::vector<NMT::Data*>& devData_,
      const int inputDim,
      const int hiddenEncDim,
      const int hiddenDim,
      const Real scale,
      const bool useBlackOut_ = false,
      const int blackOutSampleNum = 200,
      const Real blackOutAlpha = 0.4,
      const NMT::OPT opt = NMT::SGD,
      const Real clipThreshold = 3.0,
      const int beamSize = 20,
      const int maxLen = 100,
      const int miniBatchSize = 20,
      const int threadNum = 8,
      const Real learningRate = 0.5,
      const bool charDecode = false,
      const bool isTest = false,
      const int startIter = 0,
      const std::string& saveDirName = "./");
  // Corpus
  Vocabulary& sourceVoc;
  Vocabulary& targetVoc;
  std::vector<NMT::Data*>& trainData;
  std::vector<NMT::Data*>& devData;

  // Dimensional size
  int inputDim;
  int hiddenEncDim;
  int hiddenDim;

  // LSTM
  LSTM enc;
  LSTM dec;
  // Affine
  Affine stildeAffine;
  // Embeddings
  SoftMax softmax;
  BlackOut blackOut;
  MatD sourceEmbed;
  MatD targetEmbed;

  // Initialiazed vectors
  VecD zeros;
  VecD zerosEnc;

  bool useBlackOut;
  Real clipThreshold;
  Rand rnd;

  int beamSize;
  int maxLen;
  int miniBatchSize;
  int threadNum;
  Real learningRate;
  bool charDecode;
  bool isTest;
  int startIter;
  std::string saveDirName;

  /* for automated tuning */
  Real prevPerp;
  std::string prevModelFileName;
  std::string prevGradFileName;

  void encoder(const NMT::Data* data,
	       NMT::ThreadArg& arg,
	       const bool train);
  void decoder(NMT::ThreadArg& arg,
	       std::vector<LSTM::State*>& decState,
	       VecD& s_tilde,
	       const int tgtNum,
	       const int i,
	       const bool train);
  void decoderAttention(NMT::ThreadArg& arg,
			const LSTM::State* decState,
			VecD& contextSeq,
			VecD& s_tilde,
			VecD& stildeEnd);
  void decoderAttention(NMT::ThreadArg& arg,
			const int i,
			const bool train);
  void translate(NMT::Data* data,
		 NMT::ThreadArg& arg,
		 std::vector<int>& translation,
		 const bool train);
  void readStat(std::unordered_map<int, std::unordered_map<int, Real> >& stat);
  void train(NMT::Data* data,
	     NMT::ThreadArg& arg,
	     NMT::Grad& grad,
	     const bool train);
  void calculateAlphaEnc(NMT::ThreadArg& arg,
			 const LSTM::State* decState);
  void calculateAlphaEnc(NMT::ThreadArg& arg,
			 const LSTM::State* decState,
			 const int colNum);
  bool trainOpenMP(NMT::Grad& grad);
  Real calcLoss(NMT::Data* data,
		NMT::ThreadArg& arg,
		const bool train = false);
  void gradientChecking(NMT::Data* data,
			NMT::ThreadArg& arg,
			NMT::Grad& grad);
  void gradChecker(NMT::Data* data,
		   NMT::ThreadArg& arg,
		   MatD& param,
		   const MatD& grad);
  void gradChecker(NMT::Data* data,
		   NMT::ThreadArg& arg,
		   VecD& param,
		   const MatD& grad);
  void gradChecker(NMT::Data* data,
		   NMT::ThreadArg& arg,
		   NMT::Grad& grad);
  void makeTrans(const std::vector<int>& tgt,
		 std::vector<int>& trans);
  void loadCorpus(const std::string& src,
		  const std::string& tgt,
		  std::vector<NMT::Data*>& data);
  std::tuple<std::string, std::string> saveModel(NMT::Grad& grad,
						 const float i);
  void loadModel(NMT::Grad& grad,
		 const std::string& loadModelName,
		 const std::string& loadGradName);
  void saveResult(const Real value,
		  const std::string& name);
  static void demo(const std::string& srcTrain,
		   const std::string& tgtTrain,
		   const std::string& srcDev,
		   const std::string& tgtDev,
		   const int inputDim, 
		   const int hiddenEncDim,
		   const int hiddenDim,
		   const Real scale,
		   const bool useBlackOut,
		   const int blackOutSampleNum,
		   const Real blackOutAlpha,
		   const Real clipThreshold,
		   const int beamSize,
		   const int maxLen,
		   const int miniBatchSize,
		   const int threadNum,
		   const Real learningRate,
		   const int srcVocaThreshold,
		   const int tgtVocaThreshold,
		   const bool charDecode,
		   const std::string& saveDirName);
  static void demo(const std::string& srcTrain,
		   const std::string& tgtTrain,
		   const std::string& srcDev,
		   const std::string& tgtDev,
		   const int inputDim,
		   const int hiddenEncDim,
		   const int hiddenDim,
		   const Real scale,
		   const bool useBlackOut,
		   const int blackOutSampleNum,
		   const Real blackOutAlpha,
		   const Real clipThreshold,
		   const int beamSize,
		   const int maxLen,
		   const int miniBatchSize,
		   const int threadNum,
		   const Real learningRate,
		   const int srcVocaThreshold,
		   const int tgtVocaThreshold,
		   const bool charDecode,
		   const std::string& saveDirName,
		   const std::string& loadModelName,
		   const std::string& loadGradName,
		   const int startIter);
  static void evaluate(const std::string& srcTrain,
		       const std::string& tgtTrain,
		       const std::string& srcDev,
		       const std::string& tgtDev,
		       const int inputDim,
		       const int hiddenEncDim,
		       const int hiddenDim,
		       const Real scale,
		       const bool useBlackOut,
		       const int blackOutSampleNum,
		       const Real blackOutAlpha,
		       const int beamSize,
		       const int maxLen,
		       const int miniBatchSize,
		       const int threadNum,
		       const Real learningRate,
		       const int srcVocaThreshold,
		       const int tgtVocaThreshold,
		       const bool charDecode,
		       const bool istTest,
		       const std::string& saveDirName,
		       const std::string& loadModelName,
		       const std::string& loadGradName,
		       const int startIter);
  void save(const std::string& fileName);
  void load(const std::string& fileName);
};

class NMT::Data{
public:
  std::vector<int> src;
  std::vector<int> tgt;
  std::vector<int> trans; // Output of Decoder
};

class NMT::DecCandidate{
public:
  Real score;
  std::vector<int> generatedTarget;
  LSTM::State prevDec;
  LSTM::State curDec;
  std::vector<LSTM::State*> decState;
  VecD s_tilde;
  VecD stildeEnd;
  VecD contextSeq;
  VecD targetDist;
  MatD showAlphaSeq;
  bool stop;

  DecCandidate() {};
  void init(const int maxLen = 0);
};

class NMT::ThreadArg{
public:
  Rand rnd;
  // Encoder-Decoder
  std::vector<LSTM::State*> encState;
  std::vector<LSTM::State*> decState;
  // The others
  std::vector<VecD> s_tilde;
  std::vector<VecD> contextSeqList;
  std::vector<VecD> showAlphaSeq;
  std::vector<VecD> del_stilde;
  VecD del_contextSeq;
  // Affine
  std::vector<VecD> stildeEnd;
  VecD del_stildeEnd;
  // Attention Score
  MatD alphaSeq;
  VecD alphaSeqVec;
  VecD del_alphaSeq;
  VecD del_alphaSeqTmp;
  VecD del_alignScore;

  std::vector<BlackOut::State> blackOutState;
  std::vector<VecD> targetDistVec;
  VecD targetDist;

  int srcLen; // srcLen
  int tgtLen; // tgtLen
  Real loss;

  std::vector<NMT::DecCandidate> candidate; // for Beam Search

  ThreadArg() {};
  ThreadArg(NMT& nmt);
  void initTrans(const int beamSize,
		 const int maxLen);
  void initLoss();
  void init(NMT& nmt,
	    const NMT::Data* data,
	    const bool train = false);
};

class NMT::Grad{

public:
  NMT::Grad* gradHist;

  MatD sourceEmbedMatGrad;
  MatD targetEmbedMatGrad;
  std::unordered_map<int, VecD> sourceEmbed;
  std::unordered_map<int, VecD> targetEmbed;

  // LSTM
  LSTM::Grad lstmSrcGrad;
  LSTM::Grad lstmSrcRevGrad;
  LSTM::Grad lstmTgtGrad;

  // Affine
  Affine::Grad stildeAffineGrad;

  SoftMax::Grad softmaxGrad;
  BlackOut::Grad blackOutGrad;

  Grad(): gradHist(0) {}
  Grad(NMT& nmt);

  void init();
  Real norm();
  void operator += (const NMT::Grad& grad);
  void sgd(NMT& nmt,
	   const Real learningRate);
};
