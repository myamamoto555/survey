#include "Vocabulary.hpp"
#include "Utils.hpp"
#include <fstream>
#include <iostream>

struct sort_pred {
  bool operator()(const Vocabulary::Token* left, const Vocabulary::Token* right) {
    return left->count > right->count;
  }
};

Vocabulary::Vocabulary(const std::string& trainFile, const int tokenFreqThreshold){
  std::ifstream ifs(trainFile.c_str());
  std::vector<std::string> tokens;
  std::unordered_map<std::string, int> tokenCount;
  int unkCount = 0;
  int eosCount = 0;

  for (std::string line; std::getline(ifs, line); ){
    ++eosCount;
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      auto it2 = tokenCount.find(*it);

      if (it2 == tokenCount.end()){
	tokenCount[*it] = 1;
      }
      else {
	it2->second += 1;
      }
    }
  }

  for (auto it = tokenCount.begin(); it != tokenCount.end(); ++it){
    if (it->second >= tokenFreqThreshold){
      this->tokenList.push_back(new Vocabulary::Token(it->first, it->second));
    }
    else {
      unkCount += it->second;
    }
  }

  std::sort(this->tokenList.begin(), this->tokenList.end(), sort_pred());

  for (int i = 0; i < (int)this->tokenList.size(); ++i){
    this->tokenIndex[this->tokenList[i]->str] = i;
  }

  this->eosIndex = this->tokenList.size();
  this->tokenList.push_back(new Vocabulary::Token("*EOS*", eosCount));
  this->unkIndex = this->eosIndex+1;
  this->tokenList.push_back(new Vocabulary::Token("*UNK*", unkCount));
}

Vocabulary::Vocabulary(const std::string& trainFile,
		       const int tokenFreqThreshold,
		       const bool useSubword){
  std::ifstream ifs(trainFile.c_str());
  std::vector<std::string> tokens;
  std::vector<std::string> labels; // 追加
  std::unordered_map<std::string, int> tokenCount;
  int unkCount = 0;
  int eosCount = 0;

  for (std::string line; std::getline(ifs, line); ){
    ++eosCount;
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      auto it2 = tokenCount.find(*it);
      if (it2 == tokenCount.end()){
	tokenCount[*it] = 1;
      } else {
	it2->second += 1;
      }
    }
  }

  for (auto it = tokenCount.begin(); it != tokenCount.end(); ++it){
    if (it->second >= tokenFreqThreshold){
      auto subword = (it->first).find("@@");
      if (subword != std::string::npos) {
	this->tokenList.push_back(new Vocabulary::Token(it->first, it->second, true));
      } else {
	this->tokenList.push_back(new Vocabulary::Token(it->first, it->second, false));
      }
    }
    else {
      unkCount += it->second;
    }
  }

  std::sort(this->tokenList.begin(), this->tokenList.end(), sort_pred());

  for (int i = 0; i < (int)this->tokenList.size(); ++i){
    this->tokenIndex[this->tokenList[i]->str] = i;
  }
  for (auto it = labels.begin(); it != labels.end(); ++it) {
    this->depLabelIndex[this->tokenIndex[*it]] = *it; // key: tokenIndexの番号
  }

  this->eosIndex = this->tokenList.size();
  this->tokenList.push_back(new Vocabulary::Token("*EOS*", eosCount));
  this->unkIndex = this->eosIndex+1;
  this->tokenList.push_back(new Vocabulary::Token("*UNK*", unkCount));
}

Vocabulary::Vocabulary(const std::string& trainFile,
		       const int tokenFreqThreshold,
		       const char separator){
  std::ifstream ifs(trainFile.c_str());
  std::vector<std::string> tokens, components;
  std::unordered_map<std::string, int> tokenCount;
  int unkCount = 0;
  int eosCount = 0;

  for (std::string line; std::getline(ifs, line); ){
    ++eosCount;
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      auto it2 = tokenCount.find(*it);

      if (it2 == tokenCount.end()){
	tokenCount[*it] = 1;
      }
      else {
	it2->second += 1;
      }

      if (it->find(separator) != std::string::npos){
	Utils::split(*it, components, separator);

	for (auto it3 = components.begin(); it3 != components.end(); ++it3){
	  auto it4 = tokenCount.find(*it3);

	  if (it4 == tokenCount.end()){
	    tokenCount[*it3] = 1;
	  }
	  else {
	    it4->second += 1;
	  }
	}
      }
    }
  }

  for (auto it = tokenCount.begin(); it != tokenCount.end(); ++it){
    if (it->second >= tokenFreqThreshold){
      this->tokenList.push_back(new Vocabulary::Token(it->first, it->second));
    }
    else {
      unkCount += it->second;
    }
  }

  std::sort(this->tokenList.begin(), this->tokenList.end(), sort_pred());

  for (int i = 0; i < (int)this->tokenList.size(); ++i){
    this->tokenIndex[this->tokenList[i]->str] = i;
  }

  this->eosIndex = this->tokenList.size();
  this->tokenList.push_back(new Vocabulary::Token("*EOS*", eosCount));
  this->unkIndex = this->eosIndex+1;
  this->tokenList.push_back(new Vocabulary::Token("*UNK*", unkCount));
}
