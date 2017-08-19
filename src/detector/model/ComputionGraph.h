#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Utf.h"
#include "LSTM1.h"


// Each model consists of two parts, building neural graph and defining output losses.
class GraphBuilder {
public:
  std::vector<LookupNode> _target_nodes;
  std::vector<LookupNode> _tweet_nodes;
  ConcatNode _tweet_concat_node;
  ConcatNode _target_concat_node;
  BiNode _neural_output;
  AlignedMemoryPool *_pool;
  LSTM1Builder _left_to_right_target_lstm;
  LSTM1Builder _right_to_left_target_lstm;
  LSTM1Builder _left_to_right_tweet_lstm;
  LSTM1Builder _right_to_left_tweet_lstm;

  Graph *_graph;
  ModelParams *_modelParams;
  const static int max_sentence_length = 1024;

public:
  //allocate enough nodes
  void createNodes(int length_upper_bound) {

    _target_nodes.resize(length_upper_bound);
    _tweet_nodes.resize(length_upper_bound);
    _left_to_right_target_lstm.resize(length_upper_bound);
    _right_to_left_target_lstm.resize(length_upper_bound);
    _left_to_right_tweet_lstm.resize(length_upper_bound);
    _right_to_left_tweet_lstm.resize(length_upper_bound);
  }

public:
  void initial(Graph *pcg, ModelParams &model, HyperParams &opts,
    AlignedMemoryPool *mem = NULL) {
    _graph = pcg;
    for (LookupNode &n : _target_nodes) {
      n.init(opts.wordDim, opts.dropProb, mem);
      n.setParam(&model.words);
    }
    for (LookupNode &n : _tweet_nodes) {
      n.init(opts.wordDim, opts.dropProb, mem);
      n.setParam(&model.words);
    }
    _left_to_right_target_lstm.init(&model.target_left_to_right_lstm_params, opts.dropProb, true, _pool);

    _right_to_left_target_lstm.init(&model.target_right_to_left_lstm_params, opts.dropProb, false, _pool);

    _left_to_right_tweet_lstm.init(&model.tweet_left_to_right_lstm_params, opts.dropProb, true, _pool);

    _right_to_left_tweet_lstm.init(&model.tweet_right_to_left_lstm_params, opts.dropProb, false, _pool);

    _tweet_concat_node.init(opts.hiddenSize * 2, -1, mem);
    _target_concat_node.init(opts.hiddenSize * 2, -1, mem);
    _neural_output.init(opts.labelSize, -1, mem);
    _neural_output.setParam(&model.olayer_linear);
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
      _target_nodes.at(i).forward(_graph, normalizedTargetWords.at(i));
    }

    std::vector<PNode> target_nodes_ptrs;
    for (int i = 0; i < normalizedTargetWords.size(); ++i) {
      auto &n = _target_nodes.at(i);
      target_nodes_ptrs.push_back(&n);
    }

    _left_to_right_target_lstm.forward(_graph, target_nodes_ptrs);
    _right_to_left_target_lstm.forward(_graph, target_nodes_ptrs);

    _target_concat_node.forward(_graph, &_left_to_right_target_lstm._hiddens.at(normalizedTargetWords.size() - 1), &_right_to_left_target_lstm._hiddens.at(0));

    for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
      _tweet_nodes.at(i).forward(_graph, feature.m_tweet_words.at(i));
    }

    std::vector<PNode> tweet_nodes_ptrs;
    for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
      auto &n = _tweet_nodes.at(i);
      tweet_nodes_ptrs.push_back(&n);
    }

    _left_to_right_tweet_lstm.forward(_graph, tweet_nodes_ptrs);
    _right_to_left_tweet_lstm.forward(_graph, tweet_nodes_ptrs);

    _tweet_concat_node.forward(_graph, &_left_to_right_tweet_lstm._hiddens.at(feature.m_tweet_words.size() - 1), &_right_to_left_tweet_lstm._hiddens.at(0));

    _neural_output.forward(_graph, &_target_concat_node, &_tweet_concat_node);
  }
};


#endif /* SRC_ComputionGraph_H_ */
