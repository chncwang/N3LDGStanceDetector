#ifndef SRC_BASIC_TARGETS_H
#define SRC_BASIC_TARGETS_H

#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>

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
			boost::split(words, str, boost::is_any_of(" "));
			result.push_back(move(words));
		}

	return result;
}

#endif // !TARGETS_H
