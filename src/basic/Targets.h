#ifndef SRC_BASIC_TARGETS_H
#define SRC_BASIC_TARGETS_H

#include <vector>
#include <string>
#include "MyLib.h"
#include "Example.h"

enum Target {
  HILLARY = 0,
  TRUMP = 1,
  ATHEISM = 2,
  CLIMATE = 3,
  FEMINISM = 4,
  ABORTION = 5
};

const std::vector<std::string> &getTargetWords(Target target) {
  static std::vector<std::string> hillary_words = { "#hillaryclinton" };
  static std::vector<std::string> trump_words = { "#donaldtrump" };
  static std::vector<std::string> atheism_words = { "#atheism" };
  static std::vector<std::string> climate_words = { "#climatechange" };
  static std::vector<std::string> feminism_words = { "#feminism" };
  static std::vector<std::string> abortion_words = { "#prochoice" };
  static std::array<std::vector<std::string>, 6 > target_words = { hillary_words ,trump_words, atheism_words, climate_words, feminism_words, abortion_words };
  return target_words.at(target);
}

const std::vector<string> &getStanceTargets() {
  static std::vector<std::string> targets = { "Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion", "Donald Trump" };
  return targets;
}

std::vector<vector<string> > getStanceTargetWordVectors() {
  using std::move;
  auto &targets = getStanceTargets();
  std::vector<vector<string> > result;
  for (const std::string & str : targets) {
    vector<string> words;
    split_bychar(str, words);
    result.push_back(move(words));
  }

  return result;
}

bool isTargetWordInTweet(const Feature &feature) {
  std::vector<std::string> keywords;
  if (feature.m_target == Target::HILLARY) {
    keywords = { "hillary", "clinton" };
  } else if (feature.m_target == Target::TRUMP) {
    keywords = { "donald", "trump" };
  } else if (feature.m_target == Target::ATHEISM) {
    keywords = { "atheism", "atheist" };
  } else if (feature.m_target == Target::CLIMATE) {
    keywords = { "climate" };
  } else if (feature.m_target == Target::FEMINISM) {
    keywords = { "feminism", "feminist" };
  } else if (feature.m_target == Target::ABORTION) {
    keywords = { "abortion", "aborting" };
  } else {
    abort();
  }
  for (const std::string &keyword : keywords) {
    auto it = std::find(feature.m_tweet_words.begin(), feature.m_tweet_words.end(), keyword);
    if (it != feature.m_tweet_words.end()) {
      return true;
    }
  }

  return false;
}

#endif // !TARGETS_H
