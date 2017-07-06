#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"
#include "LSTM.h"
#include "MySoftMaxLoss.h"
#include "ConditionalLSTM.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet featAlpha;
	UniParams hidden_linear;
	UniParams olayer_linear; // output
	ConditionalLSTMParams lstm_params;
	ConditionalLSTMParams lstm_params2;
	//LSTMParams tweet_left_to_right_lstm_params;
	//LSTMParams tweet_right_to_left_lstm_params;
	//LSTMParams target_left_to_right_lstm_params;
	//LSTMParams target_right_to_left_lstm_params;
public:
	MySoftMaxLoss loss;

public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0){
			std::cout << "ModelParam initial - words.nVSize:" << words.nVSize << std::endl;
			abort();
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;
		opts.labelSize = 3;
		hidden_linear.initial(opts.hiddenSize, words.nDim, true, mem);
		opts.inputSize = opts.hiddenSize * 2;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
		lstm_params2.initial(opts.hiddenSize, opts.wordDim, mem);

		//tweet_left_to_right_lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
		//tweet_right_to_left_lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
		//target_left_to_right_lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
		//target_right_to_left_lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
		return true;
	}

	bool TestInitial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 ){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;
		opts.labelSize = 3;
		opts.inputSize = opts.hiddenSize * 3;
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		hidden_linear.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
		lstm_params.exportAdaParams(ada);
		lstm_params2.exportAdaParams(ada);
		//tweet_left_to_right_lstm_params.exportToAdaParams(ada);
		//tweet_right_to_left_lstm_params.exportToAdaParams(ada);
		//target_left_to_right_lstm_params.exportToAdaParams(ada);
		//target_right_to_left_lstm_params.exportToAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words E");
		//checkgrad.add(&hidden_linear.W, "hidden w");
		//checkgrad.add(&hidden_linear.b, "hidden b");
		checkgrad.add(&olayer_linear.W, "output layer W");
		//checkgrad.add(&tweet_left_to_right_lstm_params.cellParams.b, "LSTM cell b");
		//checkgrad.add(&tweet_left_to_right_lstm_params.cellParams.W1, "LSTM cell w1");
		//checkgrad.add(&tweet_left_to_right_lstm_params.cellParams.W2, "LSTM cell w2");
		//checkgrad.add(&tweet_left_to_right_lstm_params.forgetParams.W1, "LSTM forget w1");
		//checkgrad.add(&tweet_left_to_right_lstm_params.forgetParams.W2, "LSTM forget w2");
		//checkgrad.add(&tweet_left_to_right_lstm_params.forgetParams.W3, "LSTM forget w3");
		//checkgrad.add(&tweet_left_to_right_lstm_params.forgetParams.b, "LSTM forget b");
		//checkgrad.add(&tweet_left_to_right_lstm_params.inputParams.W1, "LSTM input w1");
		//checkgrad.add(&tweet_left_to_right_lstm_params.inputParams.W2, "LSTM input w2");
		//checkgrad.add(&tweet_left_to_right_lstm_params.inputParams.W3, "LSTM input w3");
		//checkgrad.add(&tweet_left_to_right_lstm_params.inputParams.b, "LSTM input b");
		//checkgrad.add(&tweet_left_to_right_lstm_params.outputParams.W1, "LSTM output w1");
		//checkgrad.add(&tweet_left_to_right_lstm_params.outputParams.W2, "LSTM output w2");
		//checkgrad.add(&tweet_left_to_right_lstm_params.outputParams.W3, "LSTM output w3");
		//checkgrad.add(&tweet_left_to_right_lstm_params.outputParams.b, "LSTM output b");
	}

	// will add it later
	void saveModel(std::ofstream &os) const{
		wordAlpha.write(os);
		words.save(os);
		hidden_linear.save(os);
		olayer_linear.save(os);
	}

	void loadModel(std::ifstream &is, AlignedMemoryPool* mem = NULL){
		wordAlpha.read(is);
		words.load(is, &wordAlpha, mem);
		hidden_linear.load(is, mem);
		olayer_linear.load(is, mem);
	}

};

#endif /* SRC_ModelParams_H_ */