#include "NMT.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <iostream>

/* Neural Machine Translation model:

   - 1-layer uni-directional Encoder-Decoder
   - Attention Mechanism (Global Attention) [Luong et al., 2015]

   - Optimizer (SGD / Momentum)

*/

#define print(var)  \
  std::cout<<(var)<<std::endl

NMT::NMT(Vocabulary& sourceVoc_,
	 Vocabulary& targetVoc_,
	 std::vector<NMT::Data*>& trainData_,
	 std::vector<NMT::Data*>& devData_,
	 const int inputDim_,
	 const int hiddenEncDim_,
	 const int hiddenDim_,
	 const Real scale,
	 const bool useBlackOut_,
	 const int blackOutSampleNum,
	 const Real blackOutAlpha,
	 const NMT::OPT opt_,
	 const Real clipThreshold_,
	 const int beamSize_,
	 const int maxLen_,
	 const int miniBatchSize_,
	 const int threadNum_,
	 const Real learningRate_,
	 const bool charDecode_,
	 const bool isTest_,
	 const int startIter_,
	 const std::string& saveDirName_):
  opt(opt_),
  sourceVoc(sourceVoc_),
  targetVoc(targetVoc_),
  trainData(trainData_),
  devData(devData_),
  inputDim(inputDim_),
  hiddenEncDim(hiddenEncDim_),
  hiddenDim(hiddenDim_),
  useBlackOut(useBlackOut_),
  clipThreshold(clipThreshold_),
  beamSize(beamSize_),
  maxLen(maxLen_),
  miniBatchSize(miniBatchSize_),
  threadNum(threadNum_),
  learningRate(learningRate_),
  charDecode(charDecode_),
  isTest(isTest_),
  startIter(startIter_),
  saveDirName(saveDirName_)
{
  // this->rnd = Rand(this->rnd.next()); // (!) TODO: For Ensemble

  // LSTM units
  this->enc = LSTM(inputDim, hiddenEncDim); // Encoder
  this->enc.init(this->rnd, scale);

  this->dec = LSTM(inputDim, hiddenDim, hiddenDim); // Decoder
  this->dec.init(this->rnd, scale);

  // LSTMs' biases set to 1
  this->enc.bf.fill(1.0);
  this->dec.bf.fill(1.0);

  // Affine
  this->stildeAffine = Affine(hiddenDim + hiddenEncDim, hiddenDim);
  this->stildeAffine.act = Affine::TANH;
  this->stildeAffine.init(this->rnd, scale);

  // Embedding matrices
  this->sourceEmbed = MatD(inputDim, this->sourceVoc.tokenList.size());
  this->targetEmbed = MatD(inputDim, this->targetVoc.tokenList.size());
  this->rnd.uniform(this->sourceEmbed, scale); // MEMO: recommended to set scale to 1.0 in Embed
  this->rnd.uniform(this->targetEmbed, scale);

  // Softmax / BlackOut
  if (!this->useBlackOut) {
    this->softmax = SoftMax(hiddenDim, this->targetVoc.tokenList.size());
  } else {
    VecD freq = VecD(this->targetVoc.tokenList.size());
    for (int i = 0; i < (int)this->targetVoc.tokenList.size(); ++i) {
      freq.coeffRef(i, 0) = this->targetVoc.tokenList[i]->count;
    }
    this->blackOut = BlackOut(hiddenDim, this->targetVoc.tokenList.size(), blackOutSampleNum);
    this->blackOut.initSampling(freq, blackOutAlpha);
  }

  this->zeros = VecD::Zero(hiddenDim); // Zero vector
  this->zerosEnc = VecD::Zero(this->hiddenEncDim); // Zero vector

  /* For automatic tuning */
  this->prevPerp = REAL_MAX;
  // this->prevModelFileName = this->saveModel(-1);
}

void NMT::encoder(const NMT::Data* data,
		  NMT::ThreadArg& arg,
		  const bool train){ // Encoder for sequence
  for (int i = 0; i < arg.srcLen; ++i) { // 入力数個
    if (i == 0) {
      this->enc.forward(this->sourceEmbed.col(data->src[i]), arg.encState[i]);
    } else {
      this->enc.forward(this->sourceEmbed.col(data->src[i]), arg.encState[i-1], arg.encState[i]);
    }
    if (train) {
      arg.encState[i]->delc = this->zerosEnc; // (!) Initialize here for backward
      arg.encState[i]->delh = this->zerosEnc;
    }
  }
}

void NMT::decoder(NMT::ThreadArg& arg,
		  std::vector<LSTM::State*>& decState,
		  VecD& s_tilde,
		  const int tgtNum,
		  const int i,
		  const bool train) {
  if (i == 0) { // initialize decoder's initial state
    arg.decState[i]->h = arg.encState[arg.srcLen-1]->h;
    arg.decState[i]->c = arg.encState[arg.srcLen-1]->c;
  } else { // i >= 1
    // input-feeding approach [Luong et al., EMNLP2015]
    this->dec.forward(this->targetEmbed.col(tgtNum), s_tilde,
		      decState[i-1], decState[i]); // (xt, at (use previous ``s_tidle``), prev, cur)
  }
  if (train) {
    arg.decState[i]->delc = this->zeros;
    arg.decState[i]->delh = this->zeros;
    arg.decState[i]->dela = this->zeros;
  }
}

void NMT::decoderAttention(NMT::ThreadArg& arg,
			   const LSTM::State* decState,
			   VecD& contextSeq,
			   VecD& s_tilde,
			   VecD& stildeEnd) { // Test
  /* Attention */
  // sequence
  contextSeq = this->zeros;

  this->calculateAlphaEnc(arg, decState);

  for (int j = 0; j < arg.srcLen; ++j) {
    contextSeq.noalias() += arg.alphaSeqVec.coeff(j, 0) * arg.encState[j]->h;
  }

  stildeEnd.segment(0, this->hiddenDim).noalias() = decState->h;
  stildeEnd.segment(this->hiddenDim, this->hiddenEncDim).noalias() = contextSeq;

  this->stildeAffine.forward(stildeEnd, s_tilde);
}

void NMT::decoderAttention(NMT::ThreadArg& arg,
			   const int i,
			   const bool train) { // Train or CalcLoss
  /* Attention */
  // sequence
  arg.contextSeqList[i] = this->zeros;

  this->calculateAlphaEnc(arg, arg.decState[i], i);

  for (int j = 0; j < arg.srcLen; ++j) {
    arg.contextSeqList[i].noalias() += arg.alphaSeq.coeff(j, i) * arg.encState[j]->h;
  }
  arg.stildeEnd[i].segment(0, this->hiddenDim).noalias() = arg.decState[i]->h;
  arg.stildeEnd[i].segment(this->hiddenDim, this->hiddenEncDim).noalias() = arg.contextSeqList[i];

  this->stildeAffine.forward(arg.stildeEnd[i], arg.s_tilde[i]);
}

struct sort_pred {
  bool operator()(const NMT::DecCandidate left, const NMT::DecCandidate right) {
    return left.score > right.score;
  }
};

void NMT::translate(NMT::Data* data,
		    NMT::ThreadArg& arg,
		    std::vector<int>& translation,
		    const bool train) {
  const Real minScore = -1.0e+05;
  const int maxLength = this->maxLen;
  const int beamSize = arg.candidate.size();
  int showNum;
  if ((int)arg.candidate.size() > 1) {
    showNum = 5;
  } else {
    showNum = 1;
  }
  MatD score = MatD(this->targetEmbed.cols(), beamSize);
  std::vector<NMT::DecCandidate> candidateTmp(beamSize);

  for (auto it = arg.candidate.begin(); it != arg.candidate.end(); ++it){
    it->init();
  }
  arg.init(*this, data, false);
  this->encoder(data, arg, false); // encoder

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      VecD stildeEnd = VecD(this->hiddenDim + this->hiddenEncDim);

      if (arg.candidate[j].stop) {
	score.col(j).fill(arg.candidate[j].score);
	continue;
      }
      if (i == 0) { // arg.candidate[j].generatedTargetのため
	arg.candidate[j].curDec.h = arg.encState[arg.srcLen-1]->h;
	arg.candidate[j].curDec.c = arg.encState[arg.srcLen-1]->c;
      } else {
	arg.candidate[j].prevDec.h = arg.candidate[j].curDec.h;
	arg.candidate[j].prevDec.c = arg.candidate[j].curDec.c;
	this->dec.forward(this->targetEmbed.col(arg.candidate[j].generatedTarget[i-1]), arg.candidate[j].s_tilde,
			  &arg.candidate[j].prevDec, &arg.candidate[j].curDec); // (xt, at (use previous ``
      }
      this->decoderAttention(arg, &arg.candidate[j].curDec, arg.candidate[j].contextSeq,
			     arg.candidate[j].s_tilde, stildeEnd);
      if (!this->useBlackOut) {
	this->softmax.calcDist(arg.candidate[j].s_tilde, arg.candidate[j].targetDist);
      } else {
	this->blackOut.calcDist(arg.candidate[j].s_tilde, arg.candidate[j].targetDist);
      }
      score.col(j).array() = arg.candidate[j].score + arg.candidate[j].targetDist.array().log(); // これまでの値と掛け算 (言語モデル評価)
    }
    for (int j = 0, row, col; j < beamSize; ++j) {
      score.maxCoeff(&row, &col); // Greedy; 最大値の要素番号を取得
      candidateTmp[j] = arg.candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop) { // if "EOS" comes up...
	score.col(col).fill(minScore);
	continue;
      }

      candidateTmp[j].generatedTarget.push_back(row); // 単語の番号追記

      if (row == this->targetVoc.eosIndex) {
	candidateTmp[j].stop = true;
      }

      if (i == 0) {
	score.row(row).fill(minScore);
      } else {
	score.coeffRef(row, col) = minScore;
      }
    }

    arg.candidate = candidateTmp; // コピー

    std::sort(arg.candidate.begin(), arg.candidate.end(), sort_pred());

    if (arg.candidate[0].generatedTarget.back() == this->targetVoc.eosIndex) { // 先頭がEOSなら終了
      break;
    }
  }

  if (train) {
    for (auto it = data->src.begin(); it != data->src.end(); ++it) {
      std::cout << this->sourceVoc.tokenList[*it]->str << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < showNum; ++i) {
      std::cout << i+1 << " (" << arg.candidate[i].score << "): ";

      for (auto it = arg.candidate[i].generatedTarget.begin(); it != arg.candidate[i].generatedTarget.end(); ++it) {
	std::cout << this->targetVoc.tokenList[*it]->str << " ";
      }
      std::cout << std::endl;
    }

    for (auto it = data->src.begin(); it != data->src.end(); ++it) {
      std::cout << this->sourceVoc.tokenList[*it]->str << " ";
    }
    std::cout << std::endl;
  } else {
    this->makeTrans(arg.candidate[0].generatedTarget, data->trans);
  }
}

void NMT::readStat(std::unordered_map<int, std::unordered_map<int, Real> >& stat) {
  std::ifstream ifs;
  if (!this->charDecode) {
    ifs.open("stat.txt", std::ios::in);
    print("Read: stat.txt.");
  } else {
    ifs.open("stat_char.txt", std::ios::in);
    print("Read: stat_char.txt.");
  }
  std::vector<std::string> res;
  VecD prob;
  int len = 0;

  for (std::string line; std::getline(ifs, line); ++len){
    Utils::split(line, res);
    prob = VecD(res.size());

    for (int i = 0; i < (int)res.size(); ++i){
      prob.coeffRef(i, 0) = atof(res[i].c_str());
    }

    if (prob.sum() == 0.0){
      continue;
    }

    for (int i = 0; i < prob.rows(); ++i){
      if (prob.coeff(i, 0) == 0.0){
	continue;
      }

      stat[len][i] = prob.coeff(i, 0);
    }
  }
}

void NMT::train(NMT::Data* data,
		NMT::ThreadArg& arg,
		NMT::Grad& grad,
		const bool train = true) { // mini-batchsize=1の学習 w/ inputFeeding
  arg.init(*this, data, train);
  this->encoder(data, arg, train); // encoder

  for (int i = 0; i < arg.tgtLen; ++i) {
    // 1) Let a decoder run forward for 1 step; PUSH
    this->decoder(arg, arg.decState, arg.s_tilde[i-1], data->tgt[i-1], i, train);
    /* Attention */
    this->decoderAttention(arg, i, train);
  }

  // Backward
  if (!this->useBlackOut) {
    for (int i = 0; i < arg.tgtLen; ++i) {
      this->softmax.calcDist(arg.s_tilde[i], arg.targetDist);
      arg.loss += this->softmax.calcLoss(arg.targetDist, data->tgt[i]);
      this->softmax.backward(arg.s_tilde[i], arg.targetDist, data->tgt[i],
			     arg.del_stilde[i], grad.softmaxGrad); // del_stildeがsoftmaxから計算される
    }
  } else { // 負例サンプル共有
    this->blackOut.sampling2(arg.blackOutState[0], this->targetVoc.unkIndex); // unk追加
    for (int i = 0; i < arg.tgtLen; ++i) {
      // word prediction
      arg.blackOutState[0].sample[0] = data->tgt[i];
      arg.blackOutState[0].weight.col(0) = this->blackOut.weight.col(data->tgt[i]);
      arg.blackOutState[0].bias.coeffRef(0, 0) = this->blackOut.bias.coeff(data->tgt[i], 0);
      this->blackOut.calcSampledDist2(arg.s_tilde[i], arg.targetDistVec[i], arg.blackOutState[0]);
      arg.loss += this->blackOut.calcSampledLoss(arg.targetDistVec[i]);
    }
    this->blackOut.backward_1(arg.tgtLen, data->tgt, arg.targetDistVec, arg.blackOutState, arg.del_stilde);
    this->blackOut.backward_2(arg.tgtLen, data->tgt, arg.s_tilde, arg.blackOutState, grad.blackOutGrad);
  }

  /* -- Backpropagation starts -- */
  for (int i = arg.tgtLen-1; i >= 0; --i) {
    if (i < arg.tgtLen-1) {
      arg.del_stilde[i].noalias() += arg.decState[i+1]->dela;
      // add gradients to the previous del_stilde
      // by input-feeding [Luong et al., EMNLP2015
    } else {}

    this->stildeAffine.backward(arg.stildeEnd[i], arg.s_tilde[i], arg.del_stilde[i],
				 arg.del_stildeEnd, grad.stildeAffineGrad);
    arg.decState[i]->delh.noalias() += arg.del_stildeEnd.segment(0, this->hiddenDim);
    // del_contextSeq
    for (int j = 0; j < arg.srcLen; ++j) { // Seq
      arg.del_alphaSeqTmp = arg.alphaSeq.coeff(j,i) * arg.del_stildeEnd.segment(this->hiddenDim, this->hiddenEncDim);
      arg.encState[j]->delh.noalias() += arg.del_alphaSeqTmp.segment(0, this->hiddenEncDim);
      arg.del_alphaSeq.coeffRef(j, 0) = arg.del_stildeEnd.segment(this->hiddenDim, this->hiddenEncDim).dot(arg.encState[j]->h);
    }
    arg.del_alignScore = arg.alphaSeq.col(i).array()*(arg.del_alphaSeq.array()-arg.alphaSeq.col(i).dot(arg.del_alphaSeq)); // X.array() - scalar; np.array() -= 1
    for (int j = 0; j < arg.srcLen; ++j) {
      arg.encState[j]->delh.noalias() += arg.del_alignScore.coeff(j, 0) * arg.decState[i]->h;
      arg.decState[i]->delh.noalias() += arg.del_alignScore.coeff(j, 0) * arg.encState[j]->h;
    }
    if (i > 0) {
      // Backward
      this->dec.backward(arg.decState[i-1], arg.decState[i],
			 grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), arg.s_tilde[i-1]);
      if (grad.targetEmbed.count(data->tgt[i-1])) {
	grad.targetEmbed.at(data->tgt[i-1]) += arg.decState[i]->delx;
      } else {
	grad.targetEmbed[data->tgt[i-1]] = arg.decState[i]->delx;
      }
    } else {}
  }

  // Decoder -> Encoder
  arg.encState[arg.srcLen-1]->delh.noalias() += arg.decState[0]->delh;
  arg.encState[arg.srcLen-1]->delc.noalias() += arg.decState[0]->delc;

  for (int i = arg.srcLen-1; i >= 0; --i) {
    if (i == 0 ) {
      this->enc.backward(arg.encState[i], grad.lstmSrcGrad,
			  this->sourceEmbed.col(data->src[i]));
    } else {
      this->enc.backward(arg.encState[i-1], arg.encState[i], grad.lstmSrcGrad,
			  this->sourceEmbed.col(data->src[i]));
    }
    if (grad.sourceEmbed.count(data->src[i])) {
      grad.sourceEmbed.at(data->src[i]).noalias() += arg.encState[i]->delx;
    } else {
      grad.sourceEmbed[data->src[i]].noalias() = arg.encState[i]->delx;
    }
  }
}

void NMT::calculateAlphaEnc(NMT::ThreadArg& arg,
			    const LSTM::State* decState) { // calculate attentional weight;
  for (int i = 0; i < arg.srcLen; ++i) {
    arg.alphaSeqVec.coeffRef(i, 0) = decState->h.dot(arg.encState[i]->h);
  }

  // softmax of ``alphaSeq``
  arg.alphaSeqVec.array() -= arg.alphaSeqVec.maxCoeff(); // stable softmax
  arg.alphaSeqVec = arg.alphaSeqVec.array().exp(); // exp() operation for all elements; np.exp(alphaSeq)
  arg.alphaSeqVec /= arg.alphaSeqVec.array().sum(); // alphaSeq.sum()
}

void NMT::calculateAlphaEnc(NMT::ThreadArg& arg,
			    const LSTM::State* decState,
			    const int colNum) { // calculate attentional weight;
  for (int i = 0; i < arg.srcLen; ++i) {
    arg.alphaSeq.coeffRef(i, colNum) = decState->h.dot(arg.encState[i]->h);
  }

  // softmax of ``alphaSeq``
  arg.alphaSeq.col(colNum).array() -= arg.alphaSeq.col(colNum).maxCoeff(); // stable softmax
  arg.alphaSeq.col(colNum) = arg.alphaSeq.col(colNum).array().exp(); // exp() operation for all elements; np.exp(alphaSeq)
  arg.alphaSeq.col(colNum) /= arg.alphaSeq.col(colNum).array().sum(); // alphaSeq.sum()
}

bool NMT::trainOpenMP(NMT::Grad& grad) { // 学習全体; Multi-threading
  static std::vector<NMT::ThreadArg> args;
  static std::vector<std::pair<int, int> > miniBatch;
  static std::vector<NMT::Grad> grads;

  Real lossTrain = 0.0;
  Real lossDev = 0.0;
  Real tgtNum = 0.0;
  Real gradNorm;
  Real lr = this->learningRate;
  static float countModel = this->startIter-0.5;

  float countModelTmp = countModel;
  std::string modelFileNameTmp = "";

  if (args.empty()) {
    grad = NMT::Grad(*this);
    for (int i = 0; i < this->threadNum; ++i) {
      args.push_back(NMT::ThreadArg(*this));
      grads.push_back(NMT::Grad(*this));
    }
    for (int i = 0, step = this->trainData.size()/this->miniBatchSize; i< step; ++i) {
      miniBatch.push_back(std::pair<int, int>(i*this->miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*this->miniBatchSize-1)));
      // Create pairs of MiniBatch, e.g. [(0,3), (4, 7), ...]
    }
  }

  auto startTmp = std::chrono::system_clock::now();
  this->rnd.shuffle(this->trainData);

  int count = 0;
  for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it) {
    std::cout << "\r"
	      << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

    for (auto it = args.begin(); it != args.end(); ++it) {
      it->initLoss();
    }
#pragma omp parallel for num_threads(this->threadNum) schedule(dynamic) shared(args, grad, grads)
    for (int i = it->first; i <= it->second; ++i) {
      int id = omp_get_thread_num();

      this->train(this->trainData[i], args[id], grads[id]);

      /* ..Gradient Checking.. :) */
      // this->gradientChecking(this->trainData[i], args[id], grads[id]);
    }
    for (int id = 0; id < this->threadNum; ++id) {
      grad += grads[id];
      grads[id].init();
      lossTrain += args[id].loss;
      args[id].loss = 0.0;
    }

    gradNorm = sqrt(grad.norm())/this->miniBatchSize;
    if (Utils::infNan2(gradNorm)) {
      countModel = countModelTmp;

      grad.init();
      std::cout << "(!) Error: INF/ NAN Gradients. Resume the training." << std::endl;

      if (modelFileNameTmp != "") {
	int systemVal = system(((std::string)"rm " + modelFileNameTmp).c_str());
	if (systemVal == -1) {
	  std::cout << "Fails to remove "
		    << modelFileNameTmp.c_str() << std::endl;
	}
      }
      return false;
    }
    lr = (gradNorm > this->clipThreshold ? this->clipThreshold*this->learningRate/gradNorm : this->learningRate);
    lr /= this->miniBatchSize;

    if (this->opt == NMT::SGD) {
      // Update the gradients by SGD
      grad.sgd(*this, lr);
    } else if (this->opt == NMT::MOMENTUM) {
    }
    grad.init();

    if (count == ((int)miniBatch.size()/2)) { // saveModel after halving epoch
      this->saveModel(grad, countModel);
      countModel += 1.;
    }

  }
  // Save a model
  std::string currentModelFileName;
  std::string currentGradFileName;
  std::tie(currentModelFileName, currentGradFileName) = this->saveModel(grad, countModel-0.5);

  std::cout << std::endl;
  auto endTmp = std::chrono::system_clock::now();
  std::cout << "Training time for this epoch: "
	    << (std::chrono::duration_cast<std::chrono::seconds>(endTmp-startTmp).count())/60.0 << "min." << std::endl;
  std::cout << "Training Loss (/sentence):    "
	    << lossTrain/this->trainData.size() << std::endl;

  startTmp = std::chrono::system_clock::now();
#pragma omp parallel for num_threads(this->threadNum)
  for (int i = 0; i < (int)this->devData.size(); ++i) {
    Real loss;
    int id = omp_get_thread_num();
    loss = this->calcLoss(this->devData[i], args[id], false);
#pragma omp critical
    {
      lossDev += loss;
      tgtNum += this->devData[i]->tgt.size();
    }
  }
  endTmp = std::chrono::system_clock::now();

  std::cout << "Evaluation time for this epoch: "
	    << (std::chrono::duration_cast<std::chrono::seconds>(endTmp-startTmp).count())/60.0
	    << "min." << std::endl;
  std::cout << "Development Perplexity and Loss (/sentence):  "
	    << exp(lossDev/tgtNum) << ", "
	    << lossDev/this->devData.size() << std::endl;

  Real devPerp = exp(lossDev/tgtNum);
  if (this->prevPerp < devPerp){
    countModel = countModelTmp;
    std::cout << "(!) Notes: Dev perplexity became worse, Resume the training!" << std::endl;

    // system(((std::string)"rm "+modelFileNameTmp).c_str());
    // system(((std::string)"rm "+currentModelFileName).c_str());

    return false;
  }
  this->prevPerp = devPerp;
  this->prevModelFileName = currentModelFileName;
  this->prevGradFileName = currentGradFileName;

  saveResult(lossTrain/this->trainData.size(), ".trainLoss"); // Training Loss
  saveResult(exp(lossDev/tgtNum), ".devPerp");                // Perplexity
  saveResult(lossDev/this->devData.size(), ".devLoss");       // Development Loss

  return true;
}

Real NMT::calcLoss(NMT::Data* data,
		   NMT::ThreadArg& arg,
		   const bool train) {
  Real loss = 0.0;

  arg.init(*this, data, false);
  this->encoder(data, arg, false);

  for (int i = 0; i < arg.tgtLen; ++i) {
    this->decoder(arg, arg.decState, arg.s_tilde[i-1], data->tgt[i-1], i, false);
    this->decoderAttention(arg, arg.decState[i], arg.contextSeqList[i], arg.s_tilde[i], arg.stildeEnd[i]);
    if (!this->useBlackOut) {
      this->softmax.calcDist(arg.s_tilde[i], arg.targetDist);
      loss += this->softmax.calcLoss(arg.targetDist, data->tgt[i]);
    } else {
      if (train) {
	// word prediction
	arg.blackOutState[0].sample[0] = data->tgt[i];
	arg.blackOutState[0].weight.col(0) = this->blackOut.weight.col(data->tgt[i]);
	arg.blackOutState[0].bias.coeffRef(0, 0) = this->blackOut.bias.coeff(data->tgt[i], 0);

	this->blackOut.calcSampledDist2(arg.s_tilde[i], arg.targetDist, arg.blackOutState[0]);
	loss += this->blackOut.calcSampledLoss(arg.targetDist); // Softmax
      } else { // Test Time
	this->blackOut.calcDist(arg.s_tilde[i], arg.targetDist); //Softmax
	loss += this->blackOut.calcLoss(arg.targetDist, data->tgt[i]); // Softmax
      }
    }
  }

  return loss;
}

void NMT::gradientChecking(NMT::Data* data,
			   NMT::ThreadArg& arg,
			   NMT::Grad& grad) {

  print("--Softmax");
  if (!this->useBlackOut) {
    print(" softmax_W");
    this->gradChecker(data, arg, this->softmax.weight, grad.softmaxGrad.weight);
    print(" softmax_b");
    this->gradChecker(data, arg, this->softmax.bias, grad.softmaxGrad.bias);
  } else {}

  // Decoder
  print("--Decoder");
  print(" stildeAffine_W");
  this->gradChecker(data, arg, this->stildeAffine.weight, grad.stildeAffineGrad.weightGrad);
  print(" stildeAffine_b");
  this->gradChecker(data, arg, this->stildeAffine.bias, grad.stildeAffineGrad.biasGrad);

  print(" dec_Wx");
  this->gradChecker(data, arg, this->dec.Wxi, grad.lstmTgtGrad.Wxi);
  this->gradChecker(data, arg, this->dec.Wxf, grad.lstmTgtGrad.Wxf);
  this->gradChecker(data, arg, this->dec.Wxo, grad.lstmTgtGrad.Wxo);
  this->gradChecker(data, arg, this->dec.Wxu, grad.lstmTgtGrad.Wxu);
  print(" dec_Wh");
  this->gradChecker(data, arg, this->dec.Whi, grad.lstmTgtGrad.Whi);
  this->gradChecker(data, arg, this->dec.Whf, grad.lstmTgtGrad.Whf);
  this->gradChecker(data, arg, this->dec.Who, grad.lstmTgtGrad.Who);
  this->gradChecker(data, arg, this->dec.Whu, grad.lstmTgtGrad.Whu);
  print(" dec_Wa");
  this->gradChecker(data, arg, this->dec.Wai, grad.lstmTgtGrad.Wai);
  this->gradChecker(data, arg, this->dec.Waf, grad.lstmTgtGrad.Waf);
  this->gradChecker(data, arg, this->dec.Wao, grad.lstmTgtGrad.Wao);
  this->gradChecker(data, arg, this->dec.Wau, grad.lstmTgtGrad.Wau);
  print(" dec_b");
  this->gradChecker(data, arg, this->dec.bi, grad.lstmTgtGrad.bi);
  this->gradChecker(data, arg, this->dec.bf, grad.lstmTgtGrad.bf);
  this->gradChecker(data, arg, this->dec.bo, grad.lstmTgtGrad.bo);
  this->gradChecker(data, arg, this->dec.bu, grad.lstmTgtGrad.bu);

  print("--Encoder");
  print(" enc_Wx");
  this->gradChecker(data, arg, this->enc.Wxi, grad.lstmSrcGrad.Wxi);
  this->gradChecker(data, arg, this->enc.Wxf, grad.lstmSrcGrad.Wxf);
  this->gradChecker(data, arg, this->enc.Wxo, grad.lstmSrcGrad.Wxo);
  this->gradChecker(data, arg, this->enc.Wxu, grad.lstmSrcGrad.Wxu);
  print(" enc_Wh");
  this->gradChecker(data, arg, this->enc.Whi, grad.lstmSrcGrad.Whi);
  this->gradChecker(data, arg, this->enc.Whf, grad.lstmSrcGrad.Whf);
  this->gradChecker(data, arg, this->enc.Who, grad.lstmSrcGrad.Who);
  this->gradChecker(data, arg, this->enc.Whu, grad.lstmSrcGrad.Whu);
  print(" enc_b");
  this->gradChecker(data, arg, this->enc.bi, grad.lstmSrcGrad.bi);
  this->gradChecker(data, arg, this->enc.bf, grad.lstmSrcGrad.bf);
  this->gradChecker(data, arg, this->enc.bo, grad.lstmSrcGrad.bo);
  this->gradChecker(data, arg, this->enc.bu, grad.lstmSrcGrad.bu);

  // Embeddings
  print("--Embedding vectors");
  print(" sourceEmbed; targetEmbed");
  this->gradChecker(data, arg, grad);
}

void NMT::gradChecker(NMT::Data* data,
		      NMT::ThreadArg& arg,
		      MatD& param,
		      const MatD& grad) {
  const Real EPS = 1.0e-04;

  for (int i = 0; i < param.rows(); ++i) {
    for (int j = 0; j < param.cols(); ++j) {
      Real val= 0.0;
      Real objFuncPlus = 0.0;
      Real objFuncMinus = 0.0;
      val = param.coeff(i, j); // Θ_i
      param.coeffRef(i, j) = val + EPS;
      objFuncPlus = this->calcLoss(data, arg, true);
      param.coeffRef(i, j) = val - EPS;
      objFuncMinus = this->calcLoss(data, arg, true);
      param.coeffRef(i, j) = val;

      Real gradVal = grad.coeff(i, j);
      Real enumVal = (objFuncPlus - objFuncMinus)/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-06) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl;
      } else {}
    }
  }
}

void NMT::gradChecker(NMT::Data* data,
		      NMT::ThreadArg& arg,
		      VecD& param,
		      const MatD& grad) {
  const Real EPS = 1.0e-04;

  for (int i = 0; i < param.rows(); ++i) {
    Real val= 0.0;
    Real objFuncPlus = 0.0;
    Real objFuncMinus = 0.0;
    val = param.coeff(i, 0); // Θ_i
    param.coeffRef(i, 0) = val + EPS;
    objFuncPlus = this->calcLoss(data, arg, true);
    param.coeffRef(i, 0) = val - EPS;
    objFuncMinus = this->calcLoss(data, arg, true);
    param.coeffRef(i, 0) = val;

    Real gradVal = grad.coeff(i, 0);
    Real enumVal = (objFuncPlus - objFuncMinus)/ (2.0*EPS);
    if ((gradVal - enumVal) > 1.0e-05) {
      std::cout << "Grad: " << gradVal << std::endl;
      std::cout << "Enum: " << enumVal << std::endl;
    } else {}
  }
}

void NMT::gradChecker(NMT::Data* data,
		      NMT::ThreadArg& arg,
		      NMT::Grad& grad) {
  const Real EPS = 1.0e-04;

  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it) {
    for (int i = 0; i < it->second.rows(); ++i) {
      Real val = 0.0;
      Real objFuncPlus = 0.0;
      Real objFuncMinus = 0.0;
      val = this->sourceEmbed.coeff(i, it->first); // クラス変数; Θ_i
      this->sourceEmbed.coeffRef(i, it->first) = val + EPS;
      objFuncPlus = this->calcLoss(data, arg, true);
      this->sourceEmbed.coeffRef(i, it->first) = val - EPS;
      objFuncMinus = this->calcLoss(data, arg, true);
      this->sourceEmbed.coeffRef(i, it->first) = val;

      Real gradVal = it->second.coeff(i, 0);
      Real enumVal = (objFuncPlus - objFuncMinus)/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-05) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl;
      } else {}
    }
  }

  for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it) {
    for (int i = 0; i < it->second.rows(); ++i) {
      Real val = 0.0;
      Real objFuncPlus = 0.0;
      Real objFuncMinus = 0.0;
      val = this->targetEmbed.coeff(i, it->first); // クラス変数; Θ_i
      this->targetEmbed.coeffRef(i, it->first) = val + EPS;
      objFuncPlus = this->calcLoss(data, arg, true);
      this->targetEmbed.coeffRef(i, it->first) = val - EPS;
      objFuncMinus = this->calcLoss(data, arg, true);
      this->targetEmbed.coeffRef(i, it->first) = val;

      Real gradVal = it->second.coeff(i, 0);
      Real enumVal = (objFuncPlus - objFuncMinus)/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-05) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl;
      } else {}
    }
  }
}

void NMT::makeTrans(const std::vector<int>& tgt,
		    std::vector<int>& trans) {
  for (auto it = tgt.begin(); it != tgt.end(); ++it) {
    if (*it != this->targetVoc.eosIndex) {
      trans.push_back(*it);
    } else {}
  }
}

void NMT::loadCorpus(const std::string& src,
		     const std::string& tgt,
		     std::vector<NMT::Data*>& data) {
  std::ifstream ifsSrc(src.c_str()); // .c_str(): 現在の文字列を返す
  std::ifstream ifsTgt(tgt.c_str());

  assert(ifsSrc);
  assert(ifsTgt);

  int numLine = 0;
  // Src
  for (std::string line; std::getline(ifsSrc, line);) {
    std::vector<std::string> tokens;
    NMT::Data *dataTmp(NULL);
    dataTmp = new NMT::Data;
    data.push_back(dataTmp);
    Utils::split(line, tokens); // 空白

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex); // 単語
    }
    data.back()->src.push_back(sourceVoc.eosIndex); // EOS
  }

  //Tgt
  for (std::string line; std::getline(ifsTgt, line);) {
    std::vector<std::string> tokens;

    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }
    data[numLine]->tgt.push_back(targetVoc.eosIndex); // EOS
    ++numLine;
  }
}

std::tuple<std::string, std::string> NMT::saveModel(NMT::Grad& grad,
						    const float i) {
  std::ostringstream oss;
  oss << this->saveDirName << "Model_NMT"
      << ".itr_" << i+1
      << ".BlackOut_" << (this->useBlackOut?"true":"false")
      << ".beamSize_" << this->beamSize
      << ".miniBatchSize_" << this->miniBatchSize
      << ".threadNum_" << this->threadNum
      << ".lrSGD_"<< this->learningRate
      << ".bin";

  this->save(oss.str());

  std::ostringstream ossGrad;
  ossGrad << this->saveDirName << "Model_NMTGrad"
	  << ".itr_" << i+1
	  << ".BlackOut_" << (this->useBlackOut?"true":"false")
	  << ".beamSize_" << this->beamSize
	  << ".miniBatchSize_" << this->miniBatchSize
	  << ".threadNum_" << this->threadNum
	  << ".lrSGD_"<< this->learningRate
	  << ".bin";
  if (this->opt == NMT::MOMENTUM) {
  } else {}

  return std::forward_as_tuple(oss.str(), ossGrad.str());
}

void NMT::loadModel(NMT::Grad& grad,
		    const std::string& loadModelName,
		    const std::string& loadGradName) {
  this->load(loadModelName.c_str());
  if (this->opt == NMT::MOMENTUM) {
  }
}

void NMT::saveResult(const Real value,
		     const std::string& name) {
  /* For Model Analysis */
  std::ofstream valueFile;
  std::ostringstream ossValue;
  ossValue << this->saveDirName << "Model_NMT" << name;

  valueFile.open(ossValue.str(), std::ios::app); // open a file with 'a' mode

  valueFile << value << std::endl;
}

void NMT::demo(const std::string& srcTrain,
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
	       const std::string& saveDirName) {

  Vocabulary sourceVoc(srcTrain, srcVocaThreshold);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold);

  std::vector<NMT::Data*> trainData;
  std::vector<NMT::Data*> devData;

  NMT nmt(sourceVoc, targetVoc, trainData, devData,
	  inputDim, hiddenEncDim, hiddenDim,
	  scale,
	  useBlackOut, blackOutSampleNum, blackOutAlpha,
	  NMT::SGD, 
	  clipThreshold,
	  beamSize, maxLen,
	  miniBatchSize, threadNum,
	  learningRate, charDecode, false,
	  0, saveDirName);

  nmt.loadCorpus(srcTrain, tgtTrain, trainData);
  nmt.loadCorpus(srcDev, tgtDev, devData);

  std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
  std::cout << "# of Development Data:\t" << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

  NMT::Grad grad(nmt);
  // Test作成
  auto test = trainData[0];
  for (int i = 0; i < 100; ++i) {
    std::cout << "\nEpoch " << i+1
	      << " (lr = " << nmt.learningRate << ")" << std::endl;

    bool status = nmt.trainOpenMP(grad);

    if (!status){
      nmt.load(nmt.prevModelFileName);
      nmt.learningRate *= 0.5;
      --i;
      continue;
    }

    // Save a model
    nmt.saveModel(grad, i);

    std::vector<NMT::ThreadArg> args;
    std::vector<std::vector<int> > translation(2);
    args.push_back(NMT::ThreadArg(nmt));
    args.push_back(NMT::ThreadArg(nmt));
    args[0].initTrans(1, nmt.maxLen);
    args[1].initTrans(5, nmt.maxLen);

    std::cout << "** Greedy Search" << std::endl;
    nmt.translate(test, args[0], translation[0], true);
    std::cout << "** Beam Search" << std::endl;
    nmt.translate(test, args[1], translation[1], true);
  }
}

void NMT::demo(const std::string& srcTrain,
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
	       const int startIter) {
  Vocabulary sourceVoc(srcTrain, srcVocaThreshold);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold);

  std::vector<NMT::Data*> trainData, devData;

  NMT nmt(sourceVoc, targetVoc,
	  trainData, devData,
	  inputDim, hiddenEncDim, hiddenDim,
	  scale,
	  useBlackOut, blackOutSampleNum, blackOutAlpha,
	  NMT::SGD, // TODO: [Check]
	  clipThreshold,
	  beamSize, maxLen,
	  miniBatchSize, threadNum,
	  learningRate, charDecode, false,
	  startIter, saveDirName);

  nmt.loadCorpus(srcTrain, tgtTrain, trainData);
  nmt.loadCorpus(srcDev, tgtDev, devData);

  std::vector<NMT::ThreadArg> args; // Evaluation of Dev.
  std::vector<std::vector<int> > translation(nmt.devData.size());
  for (int i = 0; i < threadNum; ++i){
    args.push_back(NMT::ThreadArg(nmt));
    args.back().initTrans(1, 100); // Sentences consists of less than 50 words
  }

  std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
  std::cout << "# of Development Data:\t" << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

  NMT::Grad grad(nmt);
  // Model Loaded...
  nmt.loadModel(grad, loadModelName, loadGradName);
  nmt.prevModelFileName = loadModelName;
  nmt.prevGradFileName = loadGradName;

  // Test作成
  auto test = trainData[0];

  Real lossDev = 0.;
  Real tgtNum = 0.;

#pragma omp parallel for num_threads(nmt.threadNum) schedule(dynamic) // ThreadNum
  for (int i = 0; i < (int)devData.size(); ++i) {
  int id = omp_get_thread_num();
  Real loss = nmt.calcLoss(devData[i], args[id], false);
#pragma omp critical
  {
    lossDev += loss;
    tgtNum += devData[i]->tgt.size();
  }
}

  Real currentDevPerp = exp(lossDev/tgtNum);
std::cout << "Development Perplexity and Loss (/sentence):  "
<< currentDevPerp << ", "
<< lossDev/devData.size() << "; "
<< devData.size() << std::endl;
nmt.prevPerp = currentDevPerp;

for (int i = 0; i < startIter; ++i) {
  nmt.rnd.shuffle(nmt.trainData);
 }
for (int i = startIter; i < 100; ++i) {
  std::cout << "\nEpoch " << i+1
	    << " (lr = " << nmt.learningRate
	    << ")" << std::endl;

  bool status = nmt.trainOpenMP(grad);
  if (!status){
    nmt.loadModel(grad, nmt.prevModelFileName, nmt.prevGradFileName);
    nmt.learningRate *= 0.5;
    --i;
    continue;
  }

  // Save a model
  nmt.saveModel(grad, i);

  std::vector<NMT::ThreadArg> argsTmp;
  std::vector<std::vector<int> > translation(2);
  argsTmp.push_back(NMT::ThreadArg(nmt));
  argsTmp.push_back(NMT::ThreadArg(nmt));
  argsTmp[0].initTrans(1, nmt.maxLen);
  argsTmp[1].initTrans(5, nmt.maxLen);

  std::cout << "** Greedy Search" << std::endl;
  nmt.translate(test, argsTmp[0], translation[0], true);
  std::cout << "** Beam Search" << std::endl;
  nmt.translate(test, argsTmp[1], translation[1], true);
 }
}

void NMT::evaluate(const std::string& srcTrain,
		   const std::string& tgtTrain,
		   const std::string& srcTest,
		   const std::string& tgtTest,
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
		   const bool isTest,
		   const std::string& saveDirName,
		   const std::string& loadModelName,
		   const std::string& loadGradName,
		   const int startIter) {
  static Vocabulary sourceVoc(srcTrain, srcVocaThreshold);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold);
  static std::vector<NMT::Data*> trainData, testData;

  static NMT nmt(sourceVoc, targetVoc, trainData, testData,
		 inputDim, hiddenEncDim, hiddenDim,
		 scale,
		 useBlackOut, blackOutSampleNum, blackOutAlpha,
		 NMT::SGD, 
		 3.0, // TODO: [Check] ClipThreshold
		 beamSize, maxLen,
		 miniBatchSize, threadNum,
		 learningRate, charDecode, true,
		 startIter, saveDirName);

  if (testData.empty()) {
    nmt.loadCorpus(srcTest, tgtTest, testData);
    std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
    std::cout << "# of Evaluation Data:\t" << testData.size() << std::endl;
    std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
    std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  } else {}
  std::vector<NMT::ThreadArg> args; // Evaluation of Test
  std::vector<std::vector<int> > translation(testData.size());
  for (int i = 0; i < threadNum; ++i) {
    args.push_back(NMT::ThreadArg(nmt));
    args.back().initTrans(nmt.beamSize, nmt.maxLen);
  }

  NMT::Grad grad(nmt);
  // Model Loaded...
  nmt.loadModel(grad, loadModelName, loadGradName);

  Real lossTest = 0.;
  Real tgtNum = 0.;
#pragma omp parallel for num_threads(nmt.threadNum) schedule(dynamic) shared (args) // ThreadNum
  for (int i = 0; i < (int)testData.size(); ++i) {
  Real loss;
  int id = omp_get_thread_num();
  loss = nmt.calcLoss(testData[i], args[id], false);
#pragma omp critical
  {
    lossTest += loss;
    tgtNum += testData[i]->tgt.size(); // include `*EOS*`
  }
}

  std::cout << "Perplexity and Loss (/sentence):  "
  << exp(lossTest/tgtNum) << ", "
  << lossTest/testData.size() << "; "
  << testData.size() << std::endl;

static std::unordered_map<int, std::unordered_map<int, Real> > stat; // <src, <trg, Real>>; Real = p(len(trg) | len(src))
// Load only when translate2() called for the first time
// nmt.readStat(stat);

#pragma omp parallel for num_threads(nmt.threadNum) schedule(dynamic) // ThreadNum
for (int i = 0; i < (int)testData.size(); ++i) {
auto evalData = testData[i];
int id = omp_get_thread_num();
nmt.translate(evalData, args[id], translation[i], false);
}

std::ofstream outputFile;
std::ostringstream oss;
std::string parsedMode;
oss << nmt.saveDirName << "Model_NMT"
<< ".BlackOut_" << (nmt.useBlackOut?"true":"false")
<< ".beamSize_" << nmt.beamSize
<< ".lrSGD_" << nmt.learningRate
<< ".startIter_" << startIter
<< ".charDecode_" << (nmt.charDecode?"true":"false")
<< ".Output" << (nmt.isTest?"Test":"Dev")
<< ".translate";
outputFile.open(oss.str(), std::ios::out);

for (int i = 0; i < (int)testData.size(); ++i) {
  auto evalData = testData[i];
  for (auto it = evalData->trans.begin(); it != evalData->trans.end(); ++it) {
    outputFile << nmt.targetVoc.tokenList[*it]->str << " ";
  }
  outputFile << std::endl;
  // trans
  testData[i]->trans.clear();

 }
}

void NMT::save(const std::string& fileName) {
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);
  assert(ofs);

  this->enc.save(ofs);
  this->dec.save(ofs);

  this->stildeAffine.save(ofs);

  this->softmax.save(ofs);
  this->blackOut.save(ofs);

  Utils::save(ofs, sourceEmbed);
  Utils::save(ofs, targetEmbed);
}

void NMT::load(const std::string& fileName) {
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);
  assert(ifs);

  this->enc.load(ifs);
  this->dec.load(ifs);

  this->stildeAffine.load(ifs);

  this->softmax.load(ifs);
  this->blackOut.load(ifs);

  Utils::load(ifs, sourceEmbed);
  Utils::load(ifs, targetEmbed);
}

/* NMT::DecCandidate */
void NMT::DecCandidate::init(const int maxLen) {
  this->score = 0.0;
  this->generatedTarget.clear();
  this->stop = false;

  if (this->decState.empty()) {
    for (int i = 0; i < maxLen; ++i) {
      LSTM::State *lstmDecState(NULL);
      lstmDecState = new LSTM::State;
      this->decState.push_back(lstmDecState);
    }
  }
}

/* NMT::ThreadFunc */
NMT::ThreadArg::ThreadArg(NMT& nmt) {
  // LSTM
  int stildeSize = nmt.hiddenDim + nmt.hiddenEncDim;
  for (int i = 0; i < 150; ++i) {
    LSTM::State *lstmState(NULL);
    lstmState = new LSTM::State;
    this->encState.push_back(lstmState);

    // Vectors or Matrices
    this->s_tilde.push_back(VecD(nmt.hiddenDim));
    this->contextSeqList.push_back(VecD(nmt.hiddenEncDim));
    this->del_stilde.push_back(VecD(nmt.hiddenDim));
    // Affine
    this->stildeEnd.push_back(VecD(stildeSize));
  }

  for (int i = 0; i < nmt.maxLen; ++i) {
    LSTM::State *lstmDecState(NULL);
    lstmDecState = new LSTM::State;
    this->decState.push_back(lstmDecState);
  }

  if (nmt.useBlackOut){
    for (int i = 0; i <nmt.maxLen; ++i) {
      this->blackOutState.push_back(BlackOut::State(nmt.blackOut));
      this->targetDistVec.push_back(VecD());
    }
  }
}

void NMT::ThreadArg::initTrans(const int beamSize,
			       const int maxLen) {
  for (int i = 0; i < beamSize; ++i) {
    this->candidate.push_back(NMT::DecCandidate());
    this->candidate.back().init(maxLen);
  }
}

void NMT::ThreadArg::initLoss() {
  this->loss = 0.0;
}

void NMT::ThreadArg::init(NMT& nmt,
			  const NMT::Data* data,
			  const bool train) {
  this->srcLen = data->src.size();
  this->tgtLen = data->tgt.size();

  if (train) {
    this->alphaSeq = MatD::Zero(this->srcLen, this->tgtLen);
    this->del_alphaSeq = VecD(this->srcLen);
    this->del_alphaSeqTmp = nmt.zeros;
    this->alphaSeqVec = VecD(this->srcLen);

    // for (int i = 0; i < this->srcLen; ++i) {}
    // for (int i = 0; i < this->tgtLen; ++i) {}
  } else {
    this->alphaSeqVec = VecD::Zero(this->srcLen);
  }
}

/* NMT::Grad */
NMT::Grad::Grad(NMT& nmt):
  gradHist(0)
{
  this->lstmSrcGrad = LSTM::Grad(nmt.enc);
  this->lstmTgtGrad = LSTM::Grad(nmt.dec);

  this->stildeAffineGrad = Affine::Grad(nmt.stildeAffine);

  if (!nmt.useBlackOut) {
    this->softmaxGrad = SoftMax::Grad(nmt.softmax);
  } else {
    this->blackOutGrad = BlackOut::Grad(nmt.blackOut, false);
  }

  this->init();
}

void NMT::Grad::init() {
  this->sourceEmbed.clear();
  this->targetEmbed.clear();

  this->lstmSrcGrad.init();
  this->lstmSrcRevGrad.init();
  this->lstmTgtGrad.init();

  this->stildeAffineGrad.init();

  this->softmaxGrad.init();
  this->blackOutGrad.init();
}

Real NMT::Grad::norm() {
  Real res = 0.0;

  for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
    res += it->second.squaredNorm();
  }
  for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
    res += it->second.squaredNorm();
  }

  res += this->lstmSrcGrad.norm();
  res += this->lstmSrcRevGrad.norm();
  res += this->lstmTgtGrad.norm();

  res += this->stildeAffineGrad.norm();

  res += this->softmaxGrad.norm();
  res += this->blackOutGrad.norm();

  return res;
}

void NMT::Grad::operator += (const NMT::Grad& grad) {
  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it){
    if (this->sourceEmbed.count(it->first)){
      this->sourceEmbed.at(it->first) += it->second;
    } else {
      this->sourceEmbed[it->first] = it->second;
    }
  }

  for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it){
    if (this->targetEmbed.count(it->first)){
      this->targetEmbed.at(it->first) += it->second;
    } else {
      this->targetEmbed[it->first] = it->second;
    }
  }

  this->lstmSrcGrad += grad.lstmSrcGrad;
  this->lstmSrcRevGrad += grad.lstmSrcRevGrad;
  this->lstmTgtGrad += grad.lstmTgtGrad;

  this->stildeAffineGrad += grad.stildeAffineGrad;

  this->softmaxGrad += grad.softmaxGrad;
  this->blackOutGrad += grad.blackOutGrad;
}

void NMT::Grad::sgd(NMT& nmt,
		    const Real learningRate) {
  for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it) {
    nmt.sourceEmbed.col(it->first) -= learningRate * it->second;
  }
  for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it) {
    nmt.targetEmbed.col(it->first) -= learningRate * it->second;
  }

  this->lstmSrcGrad.sgd(learningRate, nmt.enc);
  this->lstmTgtGrad.sgd(learningRate, nmt.dec);

  this->stildeAffineGrad.sgd(learningRate, nmt.stildeAffine);

  if (!nmt.useBlackOut) {
    this->softmaxGrad.sgd(learningRate, nmt.softmax);
  } else {
    nmt.blackOut.sgd(this->blackOutGrad, learningRate);
    // this->blackOutGrad.sgd(learningRate, nmt.blackOut);
  }
}
