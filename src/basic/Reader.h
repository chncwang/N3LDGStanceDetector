#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <boost/algorithm/string.hpp>
using namespace std;

#include "Instance.h"
#include "Targets.h"

vector<string> readLines(const string &fullFileName) {
  vector<string> lines;
  std::ifstream input(fullFileName);
  for (std::string line; getline(input, line);) {
    lines.push_back(line);
  }
  return lines;
}

void readLineToInstance(const string &line, Instance *instance) {
  int tailIndex = -1;
  int i = 0;
  auto targetWordVectors = getStanceTargetWordVectors();
  for (const string &target : getStanceTargets()) {
    string::size_type index = line.find(target);
    if (index <= 8) {
      string firstWord = targetWordVectors.at(i).at(0);
      if (firstWord == "Atheism") {
        instance->m_target_words = { "atheism" };
      } else if (firstWord == "Climate") {
        instance->m_target_words = { "climate", "change", "is", "a", "real", "concern" };
      } else if (firstWord == "Feminist") {
        instance->m_target_words = { "feminist" ,"movement" };
      } else if (firstWord == "Hillary") {
        instance->m_target_words = { "hillary", "clinton" };
      } else if (firstWord == "Legalization") {
        instance->m_target_words = { "legalization", "of" ,"abortion" };
      } else if (firstWord == "Donald") {
        instance->m_target_words = { "donald", "trump" };
      } else {
        std::cout << firstWord << std::endl;
        abort();
      }

      tailIndex = index + target.size();
      //cout << "Reader readLineToInstance tailIndex:" << tailIndex << endl;
      break;
    }
    ++i;
  }

  if (tailIndex == -1) {
    cout << "target not found!" << line << endl;
    assert(false);
  }

  string::size_type index = string::npos;
  for (int i = 0; i < 3; ++i) {
    Stance stance = static_cast<Stance>(i);
    const string &stanceStr = StanceToString(stance);
    //                std::cout << "stanceStr:" << stanceStr <<std::endl;
    std::regex regex(stanceStr + "\r?$");
    for (auto it = std::sregex_iterator(line.begin(), line.end(), regex);
      it != std::sregex_iterator();
      ++it) {
      index = it->position();
      instance->m_stance = stance;
      break;
    }
  }
  if (index == string::npos) {
    std::cout << line << std::endl;
    abort();
  }

  assert(index != string::npos);

  string substring = line.substr(tailIndex, index - tailIndex);
  vector<string> rawwords;
  boost::split(rawwords, substring, boost::is_any_of(" "));
  vector<string> words;
  for (string & rawword : rawwords) {
    if (rawword.empty()) continue;
    string word = normalize_to_lowerwithdigit(rawword);
    if (word == "rt" || word == "via" || word == "#semst") continue;
    if (isPunctuation(word)) continue;

    std::string http = "http";
    if (!word.compare(0, http.size(), http)) {
      continue;
    }

    assert(!word.empty());
    words.push_back(word);
  }

  for (int i = 0; i< words.size(); ++i) {
    string &word = words.at(i);
    if (word.at(0) != '#') {
      for (int j = 0; j < i; ++j) {
        string &w = words.at(j);
        if (w.at(0) == '#') {
          string symbol_removed = w.substr(1, w.size() - 1);
          swap(symbol_removed, w);
        }
      }
    }
  }

  assert(!words.empty());

  instance->m_tweet_words = move(words);
}

vector<Instance> readInstancesFromFile(const string &fullFileName) {
  vector<string> lines = readLines(fullFileName);

  vector<Instance> instances;
  using std::move;
  for (int i = 0; i < lines.size(); ++i) {
    if (lines.at(i) == "ID Target Tweet Stance") continue;
    Instance ins;
    readLineToInstance(lines.at(i), &ins);
    instances.push_back(move(ins));
  }

  return instances;
}

#endif
