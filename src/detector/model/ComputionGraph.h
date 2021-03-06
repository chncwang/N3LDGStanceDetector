#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "ConditionalLSTM.h"
#include "Utf.h"


// Each model consists of two parts, building neural graph and defining output losses.
class GraphBuilder {
public:
	vector<LookupNode> _inputNodes;
	ConditionalLSTMBuilder _left2right;
	ConditionalLSTMBuilder _right2left;
	ConcatNode _concatNode;
	UniNode _neural_output;
	AlignedMemoryPool *_pool;

  Graph *_graph;
  ModelParams *_modelParams;
  const static int max_sentence_length = 1024;

public:
  //allocate enough nodes
  void createNodes(int length_upper_bound) {
	  _inputNodes.resize(length_upper_bound);
	  _left2right.resize(length_upper_bound);
	  _right2left.resize(length_upper_bound);
  }

public:
  void initial(Graph *pcg, ModelParams &model, HyperParams &opts,
                      AlignedMemoryPool *mem = NULL) {
    _graph = pcg;
	for (LookupNode &n : _inputNodes) {
		n.init(opts.wordDim, opts.dropProb, mem);
		n.setParam(&model.words);
	}
	_left2right.init(opts.dropProb, &model.target_left_to_right_lstm_params , true, _pool);
	_right2left.init(opts.dropProb, &model.target_right_to_left_lstm_params , false, _pool);

	_concatNode.init(opts.hiddenSize * 2, -1,mem);
	_neural_output.setParam(&model.olayer_linear);
	_neural_output.init(opts.labelSize, -1, mem);
	_modelParams = &model;
	_pool = mem;
  }

public:
  // some nodes may behave different during training and decode, for example, dropout
  inline void forward(const Feature &feature, bool bTrain = false) {
    _graph->train = bTrain;

	vector<std::string> normalizedTargetWords;
	for (const std::string &w : feature.m_target_words) {
		normalizedTargetWords.push_back(normalize_to_lowerwithdigit(w));
	}

	for (int i = 0; i < normalizedTargetWords.size(); ++i) {
		_inputNodes.at(i).forward(_graph, normalizedTargetWords.at(i));
	}

	for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
		_inputNodes.at(i + normalizedTargetWords.size()).forward(_graph, feature.m_tweet_words.at(i));
	}

	vector<PNode> inputNodes;
	int totalSize = feature.m_tweet_words.size() + feature.m_target_words.size();
	for (int i = 0; i < totalSize; ++i) {
		inputNodes.push_back(&_inputNodes.at(i));
	}

	_left2right.setParam(&_modelParams->target_left_to_right_lstm_params, &_modelParams->tweet_left_to_right_lstm_params, feature.m_target_words.size());
	_right2left.setParam(&_modelParams->target_right_to_left_lstm_params, &_modelParams->tweet_right_to_left_lstm_params, feature.m_target_words.size());

	_left2right.forward(_graph, inputNodes, normalizedTargetWords.size());
	_right2left.forward(_graph, inputNodes, normalizedTargetWords.size());
	_concatNode.forward(_graph, &_left2right._hiddens.at(totalSize - 1), &_right2left._hiddens.at(normalizedTargetWords.size()));


	_neural_output.forward(_graph, &_concatNode);
  }
};


#endif /* SRC_ComputionGraph_H_ */
