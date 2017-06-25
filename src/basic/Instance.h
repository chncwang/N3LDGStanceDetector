#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
#include "Targets.h"
#include "Stance.h"

using namespace std;

class Instance
{
public:
	void clear()
	{
		m_tweet_words.clear();
		m_label.clear();
		m_sparse_feats.clear();
	}

	void evaluate(const string& predict_label, Metric& eval) const
	{
		if (predict_label == m_label)
			eval.correct_label_count++;
		eval.overall_label_count++;
	}

	void copyValuesFrom(const Instance& anInstance)
	{
		allocate(anInstance.size());
		m_label = anInstance.m_label;
		m_tweet_words = anInstance.m_tweet_words;
		m_sparse_feats = anInstance.m_sparse_feats;
		m_stance = anInstance.m_stance;
		m_target_words = anInstance.m_target_words;
	}

	int size() const {
		return m_tweet_words.size();
	}

	void allocate(int length)
	{
		clear();
		m_tweet_words.resize(length);
	}
	void assignLabel(const string &label) {
		m_label = label;
	}
public:
	vector<string> m_tweet_words;
	vector<string> m_sparse_feats;
	string m_label; //TODO
	Stance m_stance;
	const std::vector<std::string> *m_target_words;
};

#endif /*_INSTANCE_H_*/
