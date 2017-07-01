#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "LSTM.h"

class SubGraph {
public:
	vector<LookupNode> _word_inputs;
	WindowBuilder _word_window;

	LSTMBuilder _lstm_builder_left_to_right;
	LSTMBuilder _lstm_builder_right_to_left;

	Graph *_graph;

	void createNodes(int length);
	void initial(Graph *pcg, ModelParams &model, HyperParams &opts,
		AlignedMemoryPool *mem = NULL, const string &tag = "");
	void forward(const vector<string> &words);
};

void SubGraph::createNodes(int length) {
	_word_inputs.resize(length);
	_word_window.resize(length);
	_lstm_builder_left_to_right.resize(length);
	_lstm_builder_right_to_left.resize(length);
}

inline void SubGraph::initial(Graph * pcg, ModelParams & model, HyperParams & opts, AlignedMemoryPool * mem, const string &tag)
{
	_graph = pcg;
	for (int idx = 0; idx < _word_inputs.size(); idx++) {
		_word_inputs[idx].setParam(&model.words);
		_word_inputs[idx].init(opts.wordDim, mem);
	}
	_lstm_builder_left_to_right.init(&model.tweet_left_to_right_lstm_params, 0.0, true, mem, tag);
	_lstm_builder_right_to_left.init(&model.tweet_left_to_right_lstm_params, 0.0, false, mem, tag);

	_word_window.init(opts.wordDim, opts.wordContext, mem);
}

void SubGraph::forward(const vector<string> &words)
{
	int words_num = words.size();
	if (words_num <= 0) {
		abort();
	}
	if (words_num > 1024) //TODO
		words_num = 1024;

	for (int i = 0; i < words_num; i++) {
		_word_inputs[i].forward(_graph, words[i]);
	}

	vector<Node *> word_input_ptrs;
	for (LookupNode &n : _word_inputs) {
		word_input_ptrs.push_back(&n);
	}

	_word_window.forward(_graph,
		getPNodes(_word_inputs, words_num));

	vector<PNode> word_window_outputs_ptrs;
	for (ConcatNode & node : _word_window._outputs) {
		word_window_outputs_ptrs.push_back(&node);
	}
	_lstm_builder_left_to_right.forward(_graph, word_window_outputs_ptrs, words_num);
	_lstm_builder_right_to_left._shouldLeftToRight = false;
	_lstm_builder_right_to_left.forward(_graph, word_window_outputs_ptrs, words_num);
}

class ConditionalEncodingBehavior : public NodeBehavior {
public:
	void init(int outDim, AlignedMemoryPool *pool = NULL) override {}
	void forward(Graph *graph) {}
	Node &getNode() override {
		return _getNode();
	}

	std::function<Node&(void)> _getNode;
};

// Each model consists of two parts, building neural graph and defining output losses.
class GraphBuilder {
public:
	SubGraph _tweetGraph;
	SubGraph _targetGraph;

	ConcatNode _concatNode;
	LinearNode _neural_output;

  Graph *_graph;
  const static int max_sentence_length = 1024;

public:
  //allocate enough nodes
  inline void createNodes(int length_upper_bound) {
	  _tweetGraph.createNodes(length_upper_bound);
	  _targetGraph.createNodes(length_upper_bound);

	  int doubleLength = length_upper_bound * 2;
  }

public:
  void initial(Graph *pcg, ModelParams &model, HyperParams &opts,
                      AlignedMemoryPool *mem = NULL) {
    _graph = pcg;
	_tweetGraph.initial(pcg, model, opts, mem, "tweet");
	_targetGraph.initial(pcg, model, opts, mem, "target");
	_tweetGraph._lstm_builder_left_to_right._firstCellNodeBehavior = std::unique_ptr<ConditionalEncodingBehavior>(new ConditionalEncodingBehavior);
	_tweetGraph._lstm_builder_left_to_right._firstHiddenNodeBehavior = std::unique_ptr<ConditionalEncodingBehavior>(new ConditionalEncodingBehavior);
	_tweetGraph._lstm_builder_right_to_left._firstCellNodeBehavior = std::unique_ptr<ConditionalEncodingBehavior>(new ConditionalEncodingBehavior);
	_tweetGraph._lstm_builder_right_to_left._firstHiddenNodeBehavior = std::unique_ptr<ConditionalEncodingBehavior>(new ConditionalEncodingBehavior);
	_concatNode.init(opts.hiddenSize * 2, mem);
	_neural_output.setParam(&model.olayer_linear);
	_neural_output.init(opts.labelSize, mem);
  }

public:
  // some nodes may behave different during training and decode, for example, dropout
  inline void forward(const Feature &feature, bool bTrain = false) {
    _graph->train = bTrain;

	static_cast<ConditionalEncodingBehavior *>(_tweetGraph._lstm_builder_left_to_right._firstCellNodeBehavior.get())->_getNode = [&](void) ->Node& {
		return _targetGraph._lstm_builder_left_to_right._cells.at(feature.m_target_words.size() - 1);
	};

	static_cast<ConditionalEncodingBehavior *>(_tweetGraph._lstm_builder_left_to_right._firstHiddenNodeBehavior.get())->_getNode = [&](void) ->Node& {
		return _targetGraph._lstm_builder_left_to_right._hiddens.at(feature.m_target_words.size() - 1);
	};

	static_cast<ConditionalEncodingBehavior *>(_tweetGraph._lstm_builder_right_to_left._firstCellNodeBehavior.get())->_getNode = [&](void) ->Node& {
		return _targetGraph._lstm_builder_right_to_left._cells.at(0);
	};

	static_cast<ConditionalEncodingBehavior *>(_tweetGraph._lstm_builder_right_to_left._firstHiddenNodeBehavior.get())->_getNode = [&](void) ->Node& {
		return _targetGraph._lstm_builder_right_to_left._cells.at(0);
	};


	_targetGraph.forward(feature.m_target_words);
	_tweetGraph.forward(feature.m_tweet_words);

	_concatNode.forward(_graph, &_tweetGraph._lstm_builder_left_to_right._hiddens.at(feature.m_tweet_words.size() - 1), &_tweetGraph._lstm_builder_right_to_left._hiddens.at(0));
	_neural_output.forward(_graph, &_concatNode);
  }
};


#endif /* SRC_ComputionGraph_H_ */
