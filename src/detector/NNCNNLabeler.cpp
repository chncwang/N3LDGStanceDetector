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
  cout << "Creating Alphabet..." << endl;

  int numInstance;

  m_driver._modelparams.labelAlpha.clear();

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    vector<const string *> words;
	for (const string &w : *(pInstance->m_target_words)) {
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
  cout << numInstance << " " << endl;

  cout << "Label num: " << m_driver._modelparams.labelAlpha.size() << endl;
  m_driver._modelparams.labelAlpha.set_fixed_flag(true);

  return 0;
}

int Classifier::addTestAlpha(const vector<Instance> &vecInsts) {
  cout << "Adding word Alphabet..." << endl;
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

	vector<const string *> words;
	for (const string &w : *(pInstance->m_target_words)) {
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
  vector<Instance> trainInsts = readInstancesFromFile(trainFile);
  vector<Instance> devInsts = readInstancesFromFile(devFile);
  vector<Instance> testInsts = readInstancesFromFile(testFile);

  createAlphabet(trainInsts);
  addTestAlpha(devInsts);
  addTestAlpha(devInsts);

  static vector<Instance> decodeInstResults;
  static Instance curDecodeInst;
  bool bCurIterBetter = false;

  vector<Example> trainExamples, devExamples, testExamples;

  initialExamples(trainInsts, trainExamples);
  initialExamples(devInsts, devExamples);
  initialExamples(testInsts, testExamples);

  if (m_options.wordFile != "") {
    m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha,
        m_options.wordFile, m_options.wordEmbFineTune);
  } else {
    m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha,
        m_options.wordEmbSize, m_options.wordEmbFineTune);
  }

  m_driver._hyperparams.setRequared(m_options);
  m_driver.initial();

  dtype bestDIS = 0;

  int inputSize = trainExamples.size();

  int batchBlock = inputSize / m_options.batchSize;
  if (inputSize % m_options.batchSize != 0)
    batchBlock++;

  srand(0);
  std::vector<int> indexes;
  for (int i = 0; i < inputSize; ++i)
    indexes.push_back(i);

  static Metric eval, metric_dev, metric_test;
  static vector<Example> subExamples;
  int devNum = devExamples.size(), testNum = testExamples.size();
  int non_exceeds_time = 0;
  for (int iter = 0; iter < m_options.maxIter; ++iter) {
    std::cout << "##### Iteration " << iter << std::endl;

    random_shuffle(indexes.begin(), indexes.end());
    eval.reset();
    auto time_start = std::chrono::high_resolution_clock::now();
    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
      subExamples.clear();
      int start_pos = updateIter * m_options.batchSize;
      int end_pos = (updateIter + 1) * m_options.batchSize;
      if (end_pos > inputSize)
        end_pos = inputSize;

      for (int idy = start_pos; idy < end_pos; idy++) {
        subExamples.push_back(trainExamples[indexes[idy]]);
      }

      int curUpdateIter = iter * batchBlock + updateIter;
      dtype cost = m_driver.train(subExamples, curUpdateIter);

      eval.overall_label_count += m_driver._eval.overall_label_count;
      eval.correct_label_count += m_driver._eval.correct_label_count;

//      if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
//        m_driver.checkgrad(subExamples, curUpdateIter + 1);
//        std::cout << "current: " << updateIter + 1 << ", total block: "
//                  << batchBlock << std::endl;
//        std::cout << "Cost = " << cost << ", Tag Correct(%) = "
//                  << eval.getAccuracy() << std::endl;
//      }
      m_driver.updateModel();
	
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Train finished. Total time taken is: "
              << std::chrono::duration<double>(time_end - time_start).count()
              << "s" << std::endl;

    if (devNum > 0) {
      auto time_start = std::chrono::high_resolution_clock::now();
      bCurIterBetter = false;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_dev.reset();
      for (int idx = 0; idx < devExamples.size(); idx++) {
        string result_label;
        predict(devExamples[idx].m_feature, result_label);

        devInsts[idx].evaluate(Stance::NONE, metric_dev); //TODO

        if (!m_options.outBest.empty()) {
          curDecodeInst.copyValuesFrom(devInsts[idx]);
          //curDecodeInst.assignLabel(result_label);
          decodeInstResults.push_back(curDecodeInst);
        }
      }

      auto time_end = std::chrono::high_resolution_clock::now();
      std::cout << "Dev finished. Total time taken is: "
                << std::chrono::duration<double>(time_end - time_start).count()
                << "s" << std::endl;
      std::cout << "dev:" << std::endl;
      metric_dev.print();

      if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS) {
        /*m_pipe.outputAllInstances(devFile + m_options.outBest,
            decodeInstResults);*/
        bCurIterBetter = true;
      }

	  if (testNum > 0) {
		  auto time_start = std::chrono::high_resolution_clock::now();
		  if (!m_options.outBest.empty())
			  decodeInstResults.clear();
		  metric_test.reset();
		  for (int idx = 0; idx < testExamples.size(); idx++) {
			  string result_label;
			  predict(testExamples[idx].m_feature, result_label);

			  testInsts[idx].evaluate(Stance::NONE, metric_test); // TODO

			  if (bCurIterBetter && !m_options.outBest.empty()) {
				  curDecodeInst.copyValuesFrom(testInsts[idx]);
				  //curDecodeInst.assignLabel(result_label);
				  decodeInstResults.push_back(curDecodeInst);
			  }
		  }

		  auto time_end = std::chrono::high_resolution_clock::now();
		  std::cout << "Test finished. Total time taken is: "
			  << std::chrono::duration<double>(
				  time_end - time_start).count() << "s" << std::endl;
		  std::cout << "test:" << std::endl;
		  metric_test.print();

		  /*if (!m_options.outBest.empty() && bCurIterBetter) {
			  m_pipe.outputAllInstances(testFile + m_options.outBest,
				  decodeInstResults);
		  }*/
	  }

      if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS) {
        std::cout << "Exceeds best previous performance of " << bestDIS
                  << ". Saving model file.." << std::endl;
        non_exceeds_time = 0;
        bestDIS = metric_dev.getAccuracy();
        writeModelFile(modelFile);
      } else if (++non_exceeds_time > 10) {
        std::cout << "iter:" << iter << std::endl;
        break;
      }
    }
    // Clear gradients
  }
}

int Classifier::predict(const Feature &feature, string &output) {
  //assert(features.size() == words.size());
  int labelIdx;
  m_driver.predict(feature, labelIdx);
  output = m_driver._modelparams.labelAlpha.from_id(labelIdx, unknownkey);

  if (output == nullkey) {
    std::cout << "predict error" << std::endl;
  }
  return 0;
}

void Classifier::test(const string &testFile, const string &outputFile,
                      const string &modelFile) {
  loadModelFile(modelFile);
  m_driver.TestInitial();
  vector<Instance> testInsts= readInstancesFromFile(testFile);

  vector<Example> testExamples;
  initialExamples(testInsts, testExamples);

  int testNum = testExamples.size();
  vector<Instance> testInstResults;
  Metric metric_test;
  metric_test.reset();
  for (int idx = 0; idx < testExamples.size(); idx++) {
    string result_label;
    predict(testExamples[idx].m_feature, result_label);
    testInsts[idx].evaluate(Stance::NONE, metric_test); //TODO
    Instance curResultInst;
    curResultInst.copyValuesFrom(testInsts[idx]);
    //curResultInst.assignLabel(result_label);
    testInstResults.push_back(curResultInst);
  }
  std::cout << "test:" << std::endl;
  metric_test.print();

  //m_pipe.outputAllInstances(outputFile, testInstResults);
}


void Classifier::loadModelFile(const string &inputModelFile) {
  ifstream is(inputModelFile);
  if (is.is_open()) {
    m_driver._hyperparams.loadModel(is);
    m_driver._modelparams.loadModel(is, &m_driver._aligned_mem);
    is.close();
  } else
    cout << "load model error" << endl;
}

void Classifier::writeModelFile(const string &outputModelFile) {
  ofstream os(outputModelFile);
  if (os.is_open()) {
    m_driver._hyperparams.saveModel(os);
    m_driver._modelparams.saveModel(os);
    os.close();
    cout << "write model ok. " << endl;
  } else
    cout << "open output file error" << endl;
}

#include "Targets.h"

//int main(int argc, char *argv[]) {
//	vector<Instance> instances = readInstancesFromFile("C:/data/stance_data/semeval2016-task6-trainingdata.txt");
//
//	for (Instance &ins : instances) {
//		std::cout << ins.m_stance << " " << *ins.m_target << endl;
//		for (string &w : ins.m_words) {
//			std::cout << w << "|";
//		}
//		std::cout << std::endl;
//	}
//
//	while (true);
//	return 0;
//}


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
    the_classifier.test(testFile, outputFile, modelFile);
  }
  //getchar();
  //test(argv);
  //ah.write_values(std::cout);
}