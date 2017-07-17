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
		m_sparse_feats.clear();
	}

	void evaluate(Stance predict_stance, Metric& favorMetric, Metric &againstMetric) const
	{
		if (m_stance == Stance::FAVOR) {
			favorMetric.overall_label_count++;
		}
		else if (m_stance == Stance::AGAINST) {
			againstMetric.overall_label_count++;
		}

		if (predict_stance == Stance::FAVOR) {
			favorMetric.predicated_label_count++;
			if (m_stance == Stance::FAVOR) {
				favorMetric.correct_label_count++;
			}
		} else if (predict_stance == Stance::AGAINST) {
			againstMetric.predicated_label_count++;
			if (m_stance == Stance::AGAINST) {
				againstMetric.correct_label_count++;
			}
		}
	}

	void copyValuesFrom(const Instance& anInstance)
	{
		allocate(anInstance.size());
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

	std::string tostring();
public:
	vector<string> m_tweet_words;
	vector<string> m_sparse_feats;
	Stance m_stance;
	std::vector<std::string> m_target_words;
};

std::string Instance::tostring() {
	string result = "target: ";
	for (string & w : m_target_words) {
		result += w + " ";
	}
	result += "\ntweet: ";

	for (string & w : m_tweet_words) {
		result += w + " ";
	}
	result += "\nstance: " + StanceToString(m_stance);
	return result;
}

void printStanceCount(const vector<Instance> &instances) {
	int favorCount = 0;
	int againstCount = 0;
	int neutralCount = 0;
	for (const Instance &ins : instances) {
		if (ins.m_stance == Stance::FAVOR) {
			favorCount++;
		}
		else if (ins.m_stance == Stance::AGAINST) {
			againstCount++;
		}
		else if (ins.m_stance == Stance::NONE) {
			neutralCount++;
		}
		else {
			abort();
		}
	}

	std::cout << "favor: " << favorCount << " against: " << againstCount << " neutral: " << neutralCount << std::endl;
}

#endif /*_INSTANCE_H_*/
