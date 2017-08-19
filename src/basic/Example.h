#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include "Stance.h"
#include <algorithm>
#include "Targets.h"

using namespace std;

class Feature
{
public:
	vector<std::string> m_tweet_words;
	vector<std::string> m_sparse_feats;
  Target m_target;
public:
	void clear()
	{
		m_tweet_words.clear();
		m_sparse_feats.clear();
	}
};

class Example
{
public:
	Feature m_feature;
	Stance m_stance;
	//vector<dtype> m_label; //TODO

	void clear()
	{
		m_feature.clear();
	}
};

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

vector<int> getClassBalancedIndexes(const std::vector<Example> &examples) {
	std::array<std::vector<int>, 3> classSpecifiedIndexesArr;
	for (int i = 0; i < examples.size(); ++i) {
		const Example &example = examples.at(i);
		classSpecifiedIndexesArr.at(example.m_stance).push_back(i);
	}

	for (auto &v : classSpecifiedIndexesArr) {
		std::random_shuffle(v.begin(), v.end());
	}

	std::array<int, 3> counters = { classSpecifiedIndexesArr.at(0).size(), classSpecifiedIndexesArr.at(1).size(), classSpecifiedIndexesArr.at(2).size() };

	int minCounter = *std::min_element(counters.begin(), counters.end());
	std::vector<int> indexes;

	for (auto & v : classSpecifiedIndexesArr) {
		for (int i = 0; i < minCounter; ++i) {
			indexes.push_back(v.at(i));
		}
	}

	std::random_shuffle(indexes.begin(), indexes.end());
	assert(indexes.size() == 3 * minCounter);
	return indexes;
}

#endif /*_EXAMPLE_H_*/