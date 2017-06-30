#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Feature
{
public:
	vector<std::string> m_tweet_words;
	vector<std::string> m_target_words;
	vector<std::string> m_sparse_feats;
public:
	void clear()
	{
		m_tweet_words.clear();
		m_sparse_feats.clear();
		m_target_words.clear();
	}
};

class Example
{
public:
	Feature m_feature;
	vector<dtype> m_label; //TODO

	void clear()
	{
		m_feature.clear();
		m_label.clear();
	}
};

#endif /*_EXAMPLE_H_*/