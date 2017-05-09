#include "BlackOut.hpp"
#include "Utils.hpp"
#include <iostream>

void BlackOut::initSampling(const VecD& freq, const Real alpha){
  const int sum = freq.array().sum();
  const int total = sum;

  this->distWeight = freq/sum;
  this->distWeight = this->distWeight.array().pow(alpha);
  this->distWeight /= this->distWeight.sum();

  for (int i = 0; i < this->distWeight.rows(); ++i){
    for (int j = 0, num = (int)(total*this->distWeight.coeff(i, 0)); j < num; ++j){
      this->sampleDist.push_back(i);
    }
  }

  this->rnd.shuffle(this->sampleDist);
  this->distWeight = this->distWeight.array().inverse();

  this->sampleDistSize = this->sampleDist.size();
}

void BlackOut::sampling(const int label, BlackOut::State& state){
  const unsigned int SIZE = this->sampleDistSize;

  state.sample[0] = label;

  for (int i = 1, neg; i <= this->numSample; ++i){
    do {
      neg = this->sampleDist[(state.rnd.next() >> 16)%SIZE];
    } while (neg == label);

    state.sample[i] = neg;
  }
}
void BlackOut::sampling2(BlackOut::State& state, const unsigned int special){
  if (state.weight.rows() == 0){
    state.weight = MatD(this->inputDim, this->numSample+1);
    state.bias = VecD(this->numSample+1);
  }

  const unsigned int SIZE = this->sampleDistSize;
  state.sample[1] = special;
  state.weight.col(1) = this->weight.col(state.sample[1]);
  state.bias.coeffRef(1, 0) = this->bias.coeff(state.sample[1], 0);

  for (int i = 2; i <= this->numSample; ++i){
    state.sample[i] = this->sampleDist[(state.rnd.next() >> 16)%SIZE];
    state.weight.col(i) = this->weight.col(state.sample[i]);
    state.bias.coeffRef(i, 0) = this->bias.coeff(state.sample[i], 0);
  }
}

void BlackOut::calcDist(const VecD& input, VecD& output){
  output = this->bias;
  output.noalias() += this->weight.transpose()*input;
  output.array() -= output.maxCoeff(); //for numerical stability
  output = output.array().exp();
  output /= output.array().sum();
}

void BlackOut::calcSampledDist(const VecD& input, VecD& output, BlackOut::State& state){
  output = VecD(this->numSample+1);

  for (int i = 0; i < this->numSample+1; ++i){
    output.coeffRef(i, 0) =
      this->bias.coeff(state.sample[i], 0)+
      this->weight.col(state.sample[i]).dot(input);
  }

  output.array() -= output.maxCoeff();

  for (int i = 0; i < this->numSample+1; ++i){
    output.coeffRef(i, 0) =
      this->distWeight.coeff(state.sample[i], 0)*
      exp(output.coeff(i, 0));
  }

  output /= output.array().sum();
}

void BlackOut::calcSampledDist2(const VecD& input, VecD& output, BlackOut::State& state){
  output = state.bias;
  output.noalias() += state.weight.transpose()*input;
  output.array() -= output.maxCoeff();

  output.coeffRef(0, 0) =
    this->distWeight.coeff(state.sample[0], 0)*
    exp(output.coeff(0, 0));
  for (int i = 1; i < this->numSample+1; ++i){
    if (state.sample[i] == state.sample[0]){
      output.coeffRef(i, 0) = 0.0;
    }
    else {
      output.coeffRef(i, 0) =
	this->distWeight.coeff(state.sample[i], 0)*
	exp(output.coeff(i, 0));
    }
  }

  output /= output.array().sum();
}

Real BlackOut::calcLoss(const VecD& output, const int label){
  return -log(output.coeff(label, 0));
}

Real BlackOut::calcSampledLoss(const VecD& output){
  Real loss = -log(output.coeff(0, 0));

  for (int i = 1; i < output.rows(); ++i){
    loss -= log(1.0-output.coeff(i, 0));
  }

  return loss;
}

void BlackOut::backward(const VecD& input, const VecD& output, BlackOut::State& state, VecD& deltaFeature, BlackOut::Grad& grad){
  const VecD fragment = (1.0-output.block(1, 0, this->numSample, 1).array()).inverse();
  const Real sum = fragment.array().sum();
  VecD delta(this->numSample+1);

  delta.coeffRef(0, 0) = (this->numSample+1-sum)*output.coeff(0, 0)-1.0;

  for (int i = 1; i < this->numSample+1; ++i){
    delta.coeffRef(i, 0) = (this->numSample+1-(sum-fragment.coeff(i-1, 0)))*output.coeff(i, 0);
  }

  deltaFeature.noalias() = delta.coeff(0, 0)*this->weight.col(state.sample[0]);

  for (int i = 1; i < this->numSample+1; ++i){
    deltaFeature.noalias() += delta.coeff(i, 0)*this->weight.col(state.sample[i]);
  }

  for (int i = 0; i < this->numSample+1; ++i){
    if (grad.bias.count(state.sample[i])){
      grad.weight.at(state.sample[i]).noalias() += delta.coeff(i, 0)*input;
      grad.bias.at(state.sample[i]) += delta.coeff(i, 0);
    }
    else {
      grad.weight[state.sample[i]].noalias() = delta.coeff(i, 0)*input;
      grad.bias[state.sample[i]] = delta.coeff(i, 0);
    }
  }
}

void BlackOut::backward_(const VecD& input, const VecD& output, const unsigned int label, BlackOut::State& state, VecD& deltaFeature, BlackOut::Grad& grad){
  state.fragment = (1.0-output.segment(1, this->numSample).array()).inverse();
  const Real sum = state.fragment.array().sum();

  state.delta.coeffRef(0, 0) = (this->numSample+1-sum)*output.coeff(0, 0)-1.0;
  for (int i = 1; i < this->numSample+1; ++i){
    state.delta.coeffRef(i, 0) = (this->numSample+1-(sum-state.fragment.coeff(i-1, 0)))*output.coeff(i, 0);
  }

  deltaFeature.noalias() = state.delta.coeff(0, 0)*state.weight.col(0);
  for (int i = 1; i < this->numSample+1; ++i){
    deltaFeature.noalias() += state.delta.coeff(i, 0)*state.weight.col(i);
  }

  for (int i = 0; i < this->numSample+1; ++i){
    if (grad.bias.count(state.sample[i])){
      grad.weight.at(state.sample[i]).noalias() += state.delta.coeff(i, 0)*input;
      grad.bias.at(state.sample[i]) += state.delta.coeff(i, 0);
    }
    else {
      grad.weight[state.sample[i]].noalias() = state.delta.coeff(i, 0)*input;
      grad.bias[state.sample[i]] = state.delta.coeff(i, 0);
    }
  }
}

void BlackOut::backward_1(const unsigned int len, const std::vector<int>& label, const std::vector<VecD>& output, std::vector<BlackOut::State>& state, std::vector<VecD>& deltaFeature){ // TODO: originalはconst std::vector<unsigned int>& label
  for (unsigned int j = 0; j < len; ++j){
    state[0].fragment = (1.0-output[j].segment(1, this->numSample).array()).inverse();
    const Real sum = state[0].fragment.array().sum();

    state[j].delta.coeffRef(0, 0) = (this->numSample+1-sum)*output[j].coeff(0, 0)-1.0;

    for (int i = 1; i < this->numSample+1; ++i){
      state[j].delta.coeffRef(i, 0) = (this->numSample+1-(sum-state[0].fragment.coeff(i-1, 0)))*output[j].coeff(i, 0);
    }
    deltaFeature[j] = state[j].delta.coeff(0, 0)*this->weight.col(label[j]);
  }
  for (int i = 1; i < this->numSample+1; ++i){
    for (unsigned int j = 0; j < len; ++j){
      deltaFeature[j].noalias() += state[j].delta.coeff(i, 0)*state[0].weight.col(i);
    }
  }
}
void BlackOut::backward_2(const unsigned int len, const std::vector<int>& label, const std::vector<VecD>& input, std::vector<BlackOut::State>& state, BlackOut::Grad& grad){
  for (unsigned int j = 0; j < len; ++j){
    if (grad.bias.count(label[j])){
      grad.weightMat.col(label[j]) += state[j].delta.coeff(0, 0)*input[j];
      grad.bias.at(label[j]) += state[j].delta.coeff(0, 0);
    }
    else {
      grad.weightMat.col(label[j]) = state[j].delta.coeff(0, 0)*input[j];
      grad.bias[label[j]] = state[j].delta.coeff(0, 0);
    }
  }

  for (int i = 1; i < this->numSample+1; ++i){
    for (unsigned int j = 0; j < len; ++j){
      if (grad.bias.count(state[0].sample[i])){
	grad.weightMat.col(state[0].sample[i]) += state[j].delta.coeff(i, 0)*input[j];
	grad.bias.at(state[0].sample[i]) += state[j].delta.coeff(i, 0);
      }
      else {
	grad.weightMat.col(state[0].sample[i]) = state[j].delta.coeff(i, 0)*input[j];
	grad.bias[state[0].sample[i]] = state[j].delta.coeff(i, 0);
      }
    }
  }
}

void BlackOut::backward1(const unsigned int len, const std::vector<VecD>& output, std::vector<BlackOut::State>& state, std::vector<VecD>& deltaFeature){
  for (unsigned int j = 0; j < len; ++j){
    state[j].fragment = (1.0-output[j].segment(1, this->numSample).array()).inverse();
    const Real sum = state[j].fragment.array().sum();

    state[j].delta.coeffRef(0, 0) = (this->numSample+1-sum)*output[j].coeff(0, 0)-1.0;

    for (int i = 1; i < this->numSample+1; ++i){
      state[j].delta.coeffRef(i, 0) = (this->numSample+1-(sum-state[j].fragment.coeff(i-1, 0)))*output[j].coeff(i, 0);
    }
    deltaFeature[j] = VecD::Zero(this->inputDim);
  }
  std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, unsigned int> > > mp;
  for (unsigned int i = 0; i < len; ++i){
    for (unsigned int j = 0; j < state[i].sample.size(); ++j){
      mp[state[i].sample[j]].push_back(std::pair<unsigned int, unsigned int>(i, j));
    }
  }
  for (unsigned int i = 0; i < this->outputDim; ++i){
    auto it = mp.find(i);
    if (it == mp.end()){
      continue;
    }
    for (auto itit = it->second.begin(); itit != it->second.end(); ++itit){
      deltaFeature[itit->first] += state[itit->first].delta.coeff(itit->second, 0)*this->weight.col(it->first);
    }
  }
}

void BlackOut::backward2(const unsigned int len, const std::vector<VecD>& input, std::vector<BlackOut::State>& state, BlackOut::Grad& grad){
  std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, unsigned int> > > mp;
  for (unsigned int i = 0; i < len; ++i){
    for (unsigned int j = 0; j < state[i].sample.size(); ++j){
      mp[state[i].sample[j]].push_back(std::pair<unsigned int, unsigned int>(i, j));
    }
  }
  for (unsigned int i = 0; i < this->outputDim; ++i){
    auto it = mp.find(i);
    if (it == mp.end()){
      continue;
    }
    for (auto itit = it->second.begin(); itit != it->second.end(); ++itit){
      if (grad.bias.count(it->first)){
	if (grad.useMap){
	  grad.weight.at(it->first).noalias() += state[itit->first].delta.coeff(itit->second, 0)*input[itit->first];
	}
	else {
	  grad.weightMat.col(it->first).noalias() += state[itit->first].delta.coeff(itit->second, 0)*input[itit->first];
	}
	grad.bias.at(it->first) += state[itit->first].delta.coeff(itit->second, 0);
      }
      else {
	if (grad.useMap){
	  grad.weight[it->first].noalias() = state[itit->first].delta.coeff(itit->second, 0)*input[itit->first];
	}
	else {
	  grad.weightMat.col(it->first).noalias() = state[itit->first].delta.coeff(itit->second, 0)*input[itit->first];
	}
	grad.bias[it->first] = state[itit->first].delta.coeff(itit->second, 0);
      }
    }
  }
}

void BlackOut::backward(const VecD& input, const VecD& output, const Real scale, BlackOut::State& state, VecD& deltaFeature, BlackOut::Grad& grad){
  const VecD fragment = (1.0-output.block(1, 0, this->numSample, 1).array()).inverse();
  const Real sum = fragment.array().sum();
  VecD delta(this->numSample+1);

  delta.coeffRef(0, 0) = (this->numSample+1-sum)*output.coeff(0, 0)-1.0;

  for (int i = 1; i < this->numSample+1; ++i){
    delta.coeffRef(i, 0) = (this->numSample+1-(sum-fragment.coeff(i-1, 0)))*output.coeff(i, 0);
  }

  delta.array() *= scale;

  deltaFeature.noalias() = delta.coeff(0, 0)*this->weight.col(state.sample[0]);

  for (int i = 1; i < this->numSample+1; ++i){
    deltaFeature.noalias() += delta.coeff(i, 0)*this->weight.col(state.sample[i]);
  }

  for (int i = 0; i < this->numSample+1; ++i){
    if (grad.bias.count(state.sample[i])){
      grad.weight.at(state.sample[i]).noalias() += delta.coeff(i, 0)*input;
      grad.bias.at(state.sample[i]) += delta.coeff(i, 0);
    }
    else {
      grad.weight[state.sample[i]].noalias() = delta.coeff(i, 0)*input;
      grad.bias[state.sample[i]] = delta.coeff(i, 0);
    }
  }
}

void BlackOut::sgd(const BlackOut::Grad& grad, const Real learningRate){
  for (auto it = grad.weight.begin(); grad.useMap && it != grad.weight.end(); ++it){
    this->weight.col(it->first) -= learningRate*it->second;
  }

  for (auto it = grad.bias.begin(); it != grad.bias.end(); ++it){
    if (!grad.useMap){
      this->weight.col(it->first) -= learningRate*grad.weightMat.col(it->first);
    }
    this->bias.coeffRef(it->first, 0) -= learningRate*it->second;
  }
}

void BlackOut::save(std::ofstream& ofs){
  Utils::save(ofs, this->weight);
  Utils::save(ofs, this->bias);
}

void BlackOut::load(std::ifstream& ifs){
  Utils::load(ifs, this->weight);
  Utils::load(ifs, this->bias);
}

void BlackOut::Grad::init(){
  this->weight.clear();
  this->bias.clear();
}

Real BlackOut::Grad::norm(){
  Real res = 0.0;

  for (auto it = this->weight.begin(); this->useMap && it != this->weight.end(); ++it){
    res += it->second.squaredNorm();
  }
  for (auto it = this->bias.begin(); it != this->bias.end(); ++it){
    if (!this->useMap){
      res += this->weightMat.col(it->first).squaredNorm();
    }
    res += it->second*it->second;
  }

  return res;
}

void BlackOut::Grad::sgd(const Real learningRate, BlackOut& blackout){
  for (auto it = this->weight.begin(); this->useMap && it != this->weight.end(); ++it){
    blackout.weight.col(it->first) -= learningRate*it->second;
  }

  for (auto it = this->bias.begin(); it != this->bias.end(); ++it){
    if (!this->useMap){
      blackout.weight.col(it->first) -= learningRate*this->weightMat.col(it->first);
    }
    blackout.bias.coeffRef(it->first, 0) -= learningRate*it->second;
  }
}

void BlackOut::Grad::momentum(const Real learningRate, const Real m,  BlackOut& blackout){
  if (this->gradHist == 0){
    this->gradHist = new BlackOut::Grad();
    this->gradHist->weightMat = MatD::Zero(blackout.weight.rows(), blackout.weight.cols());
    this->gradHist->biasVec = VecD::Zero(blackout.bias.rows());
  }

  for (auto it = this->bias.begin(); it != this->bias.end(); ++it){
    this->gradHist->weightMat.col(it->first).array() *= m;
    this->gradHist->biasVec.coeffRef(it->first, 0) *= m;
    this->gradHist->weightMat.col(it->first) += learningRate*this->weightMat.col(it->first);
    this->gradHist->biasVec.coeffRef(it->first, 0) += learningRate*it->second;
    blackout.weight.col(it->first) -= this->gradHist->weightMat.col(it->first);
    blackout.bias.coeffRef(it->first, 0) -= this->gradHist->biasVec.coeff(it->first, 0);
  }
}

void BlackOut::Grad::saveHist(std::ofstream& ofs){
  Utils::save(ofs, this->gradHist->weightMat);
  Utils::save(ofs, this->gradHist->biasVec);
}

void BlackOut::Grad::loadHist(std::ifstream& ifs){
  Utils::load(ifs, this->gradHist->weightMat);
  Utils::load(ifs, this->gradHist->biasVec);
}

void BlackOut::Grad::operator += (const BlackOut::Grad& grad){
  for (auto it = grad.weight.begin(); this->useMap && it != grad.weight.end(); ++it){
    if (this->weight.count(it->first)){
      this->weight.at(it->first) += it->second;
    }
    else {
      this->weight[it->first] = it->second;
    }
  }

  for (auto it = grad.bias.begin(); it != grad.bias.end(); ++it){
    if (this->bias.count(it->first)){
      if (!this->useMap){
	this->weightMat.col(it->first) += grad.weightMat.col(it->first);
      }
      this->bias.at(it->first) += it->second;
    }
    else {
      if (!this->useMap){
	this->weightMat.col(it->first) = grad.weightMat.col(it->first);
      }
      this->bias[it->first] = it->second;
    }
  }
}

void BlackOut::Grad::operator /= (const Real val){
  for (auto it = this->weight.begin(); it != this->weight.end(); ++it){
    it->second /= val;
  }
  for (auto it = this->bias.begin(); it != this->bias.end(); ++it){
    it->second /= val;
  }
}
