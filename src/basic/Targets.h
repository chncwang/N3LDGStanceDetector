#ifndef SRC_BASIC_TARGETS_H
#define SRC_BASIC_TARGETS_H

#include <vector>
#include <string>
#include "MyLib.h"
#include "Example.h"

const std::vector<string> &getStanceTargets() {
	static std::vector<std::string> targets = {"Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion", "Donald Trump"};
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
	const std::string &firstWord = feature.m_target_words.at(0);
	std::vector<std::string> keywords;
	if (firstWord == "#hillaryclinton") {
		keywords = {"hillary", "clinton"};
	}
	else if (firstWord == "#donaldtrump") {
		keywords = {"donald", "trump"};
	}
	else if (firstWord == "#atheism") {
		keywords = { "atheism", "atheist" };
	}
	else if (firstWord == "#climatechange") {
		keywords = {"climate"};
	}
	else if (firstWord == "#feminism") {
		keywords = {"feminism", "feminist"};
	}
	else if (firstWord == "#prochoice") {
		keywords = {"abortion", "aborting"};
	}
	else {
		abort();
	}
	for (const std::string &keyword : keywords) {
		auto it =std::find(feature.m_tweet_words.begin(), feature.m_tweet_words.end(), keyword);
		if (it != feature.m_tweet_words.end()) {
			return true;
		}
	}

	return false;
}

#endif // !TARGETS_H
