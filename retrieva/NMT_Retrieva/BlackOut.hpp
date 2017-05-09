// まだコードは公開してない
#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include <vector>
#include <unordered_map>
#include <fstream>

class BlackOut{
public:
  BlackOut():
    numSample(1)
  {}
  BlackOut(const int inputDim_, const int classNum, const int numSample_):
    weight(MatD::Zero(inputDim_, classNum)), bias(VecD::Zero(classNum)),
    numSample(numSample_),
    inputDim(inputDim_),
    outputDim(classNum)
  {}

  class State;
  class Grad;

  Rand rnd;
  MatD weight; VecD bias;
  int numSample;
  std::vector<int> sampleDist;
  VecD distWeight;

  unsigned int sampleDistSize;
  unsigned int inputDim;
  unsigned int outputDim;

  void initSampling(const VecD& freq, const Real alpha);
  void sampling(const int label, BlackOut::State& state);
  void sampling2(BlackOut::State& state, const unsigned int special);
  void calcDist(const VecD& input, VecD& output);
  void calcSampledDist(const VecD& input, VecD& output, BlackOut::State& state);
  void calcSampledDist2(const VecD& input, VecD& output, BlackOut::State& state);
  Real calcLoss(const VecD& output, const int label);
  Real calcSampledLoss(const VecD& output);
  void backward(const VecD& input, const VecD& output, BlackOut::State& state, VecD& deltaFeature, BlackOut::Grad& grad);
  void backward_(const VecD& input, const VecD& output, const unsigned int label, BlackOut::State& state, VecD& deltaFeature, BlackOut::Grad& grad);
  void backward_1(const unsigned int len, const std::vector<int>& label, const std::vector<VecD>& output, std::vector<BlackOut::State>& state, std::vector<VecD>& deltaFeature);
  void backward_2(const unsigned int len, const std::vector<int>& label, const std::vector<VecD>& input, std::vector<BlackOut::State>& state, BlackOut::Grad& grad);
  void backward1(const unsigned int len, const std::vector<VecD>& output, std::vector<BlackOut::State>& state, std::vector<VecD>& deltaFeature);
  void backward2(const unsigned int len, const std::vector<VecD>& input, std::vector<BlackOut::State>& state, BlackOut::Grad& grad);
  void backward(const VecD& input, const VecD& output, const Real scale, BlackOut::State& state, VecD& deltaFeature, BlackOut::Grad& grad);

  void sgd(const BlackOut::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

  class BlackOut::State{
  public:
    State(){};
    State(BlackOut& blackout):
      rnd(Rand(blackout.rnd.next())),
      sample(std::vector<int>(blackout.numSample+1)),
      delta(VecD(blackout.numSample+1))
    {};

    Rand rnd;
    std::vector<int> sample;
    MatD weight;
    VecD bias;

    VecD fragment;
    VecD delta;
  };

    class BlackOut::Grad{
    public:
      Grad(): gradHist(0) {}
      Grad(const BlackOut& blackout, const bool useMap_ = true):
	gradHist(0)
      {if (!useMap_){this->weightMat = blackout.weight;} this->useMap = useMap_;}

      BlackOut::Grad* gradHist;

      bool useMap;

      MatD weightMat;
      VecD biasVec;
      std::unordered_map<int, VecD> weight;
      std::unordered_map<int, Real> bias;

      void init();
      Real norm();
      void sgd(const Real learningRate, BlackOut& blackout);
      void adagrad(const Real learningRate, BlackOut& blackOut, const Real initVal = 1.0);
      void momentum(const Real learningRate, const Real m, BlackOut& blackout);
      void saveHist(std::ofstream& ofs);
      void loadHist(std::ifstream& ifs);

      void operator += (const BlackOut::Grad& grad);
      void operator /= (const Real val);
    };
