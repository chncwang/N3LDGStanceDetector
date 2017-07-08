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
	vector<ConcatNode> _lstmConcatNodes;
	AvgPoolNode _avg_pooling;
	MaxPoolNode _max_pooling;
	MinPoolNode _min_pooling;
	ConcatNode _poolingConcatNode;
	LinearNode _neural_output;
	float _dropout;
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
	  _lstmConcatNodes.resize(length_upper_bound);
	  _avg_pooling.setParam(length_upper_bound);
	  _max_pooling.setParam(length_upper_bound);
	  _min_pooling.setParam(length_upper_bound);
  }

public:
  void initial(Graph *pcg, ModelParams &model, HyperParams &opts,
                      AlignedMemoryPool *mem = NULL) {
    _graph = pcg;
	for (LookupNode &n : _inputNodes) {
		n.init(opts.wordDim, opts.dropProb, mem);
		n.setParam(&model.words);
	}
	_left2right.init(_dropout, &model.target_left_to_right_lstm_params , true, _pool);
	_right2left.init(_dropout, &model.target_right_to_left_lstm_params , false, _pool);

	_dropout = opts.dropProb;
	int doubleHiddenSize = 2 * opts.hiddenSize;
	for (ConcatNode &n : _lstmConcatNodes) {
		n.init(doubleHiddenSize, -1,mem);
	}

	_avg_pooling.init(doubleHiddenSize, -1, mem);
	_max_pooling.init(doubleHiddenSize, -1, mem);
	_min_pooling.init(doubleHiddenSize, -1, mem);

	_poolingConcatNode.init(doubleHiddenSize * 3, -1, mem);

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

	for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
		_lstmConcatNodes.at(i).forward(_graph, &_left2right._hiddens.at(feature.m_target_words.size() + i), &_right2left._hiddens.at(feature.m_target_words.size() + i));
	}

	_avg_pooling.forward(_graph, getPNodes(_lstmConcatNodes, feature.m_tweet_words.size()));
	_min_pooling.forward(_graph, getPNodes(_lstmConcatNodes, feature.m_tweet_words.size()));
	_max_pooling.forward(_graph, getPNodes(_lstmConcatNodes, feature.m_tweet_words.size()));

	_poolingConcatNode.forward(_graph, &_avg_pooling, &_min_pooling, &_max_pooling);

	_neural_output.forward(_graph, &_poolingConcatNode);
  }
};


#endif /* SRC_ComputionGraph_H_ */
