#include "NNCNNLabeler.h"
#include "Stance.h"

#include <chrono> 
#include "Argument_helper.h"
#include "Reader.h"

Classifier::Classifier(int memsize) : m_driver(memsize) {
  srand(0);
}

Classifier::~Classifier() {
}

int Classifier::createAlphabet(const vector<Instance> &vecInsts) {
  if (vecInsts.size() == 0) {
    std::cout << "training set empty" << std::endl;
    return -1;
  }
  std::cout << "Creating Alphabet..." << endl;

  int numInstance;

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    vector<const string *> words;
	for (const string &w : pInstance->m_target_words) {
		words.push_back(&w);
	}

	for (const string &w : pInstance->m_tweet_words) {
		words.push_back(&w);
	}

	for (const string *w : words) {
		string normalizedWord = normalize_to_lowerwithdigit(*w);

		if (m_word_stats.find(normalizedWord) == m_word_stats.end()) {
			m_word_stats.insert(std::pair<std::string, int>(normalizedWord, 0));
		}
		else {
			m_word_stats.at(normalizedWord) += 1;
		}
	}

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }

    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }
  std::cout << numInstance << " " << endl;

  return 0;
}

int Classifier::addTestAlpha(const vector<Instance> &vecInsts) {
  std::cout << "Adding word Alphabet..." << endl;
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

	vector<const string *> words;
	for (const string &w : pInstance->m_target_words) {
		words.push_back(&w);
	}

	for (const string &w : pInstance->m_tweet_words) {
		words.push_back(&w);
	}

	for (const string *w : words) {
		string normalizedWord = normalize_to_lowerwithdigit(*w);

		if (m_word_stats.find(normalizedWord) == m_word_stats.end()) {
			m_word_stats.insert(std::pair<std::string, int>(normalizedWord, 0));
		}
		else {
			m_word_stats.at(normalizedWord) += 1;
		}
	}

    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }
  cout << numInstance << " " << endl;

  return 0;
}


void Classifier::extractFeature(Feature &feat, const Instance *pInstance) {
  feat.m_tweet_words = pInstance->m_tweet_words;
  feat.m_target_words = pInstance->m_target_words;
  feat.m_sparse_feats = pInstance->m_sparse_feats;
}

void Classifier::convert2Example(const Instance *pInstance, Example &exam) {
	vector<dtype> stanceVector = { 0, 0, 0 };
	stanceVector.at(pInstance->m_stance) = 1;
	exam.m_label = stanceVector;
	Feature feature;
	extractFeature(feature, pInstance);
	exam.m_feature = feature;
}

void Classifier::initialExamples(const vector<Instance> &vecInsts,
                                 vector<Example> &vecExams) {
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
	  const Instance *pInstance = &vecInsts[numInstance];
	  Example curExam;
	  convert2Example(pInstance, curExam);
	  vecExams.push_back(curExam);
  }
}

void Classifier::train(const string &trainFile, const string &devFile,
                       const string &testFile, const string &modelFile,
                       const string &optionFile) {
  if (optionFile != "")
    m_options.load(optionFile);
  m_options.showOptions();

  vector<Instance> rawtrainInsts = readInstancesFromFile(trainFile);
  vector<Instance> trainInsts;
  for (Instance &ins : rawtrainInsts) {
	  if (ins.m_target_words.at(0) == "#hillaryclinton") {
		  continue;
	  }
	  trainInsts.push_back(ins);
  }

  std::cout << "train set:" << std::endl;
  for (Instance &ins : trainInsts) {
	  std::cout << ins.tostring() << std::endl;
  }

  std::cout << "dev set:" << std::endl;
  vector<Instance> devInsts = readInstancesFromFile(devFile);
  for (Instance &ins : devInsts) {
	  std::cout << ins.tostring() << std::endl;
  }

  std::cout << "test set:" << std::endl;
  vector<Instance> testInsts = readInstancesFromFile(testFile);
  for (Instance &ins : testInsts) {
	  std::cout << ins.tostring() << std::endl;
  }
}


void Classifier::loadModelFile(const string &inputModelFile) {
  ifstream is(inputModelFile);
  if (is.is_open()) {
    m_driver._hyperparams.loadModel(is);
    m_driver._modelparams.loadModel(is, &m_driver._aligned_mem);
    is.close();
  } else
    std::cout << "load model error" << endl;
}

void Classifier::writeModelFile(const string &outputModelFile) {
  ofstream os(outputModelFile);
  if (os.is_open()) {
    m_driver._hyperparams.saveModel(os);
    m_driver._modelparams.saveModel(os);
    os.close();
    std::cout << "write model ok. " << endl;
  } else
    std::cout << "open output file error" << endl;
}

int main(int argc, char *argv[]) {
  std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
  std::string outputFile = "";
  bool bTrain = false;
  int memsize = 0;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("train", "trainCorpus", "named_string",
      "training corpus to train a model, must when training", trainFile);
  ah.new_named_string("dev", "devCorpus", "named_string",
      "development corpus to train a model, optional when training", devFile);
  ah.new_named_string("test", "testCorpus", "named_string",
      "testing corpus to train a model or input file to test a model, optional when training and must when testing",
      testFile);
  ah.new_named_string("model", "modelFile", "named_string",
      "model file, must when training and testing", modelFile);
  ah.new_named_string("option", "optionFile", "named_string",
      "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string",
      "output file to test, must when testing", outputFile);
  ah.new_named_int("memsize", "memorySize", "named_int",
      "This argument decides the size of static memory allocation", memsize);

  ah.process(argc, argv);

  if (memsize < 0)
    memsize = 0;
  Classifier the_classifier(memsize);
  if (bTrain) {
    the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
  } else {
  }
  //getchar();
  //test(argv);
  //ah.write_values(std::cout);
}
