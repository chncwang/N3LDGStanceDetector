#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"
#include "MySoftMaxLoss.h"
#include "LSTM1.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{
public:
	LookupTable words; // should be initialized outside
	Alphabet wordAlpha; // should be initialized outside
	Alphabet featAlpha;
	UniParams hidden_linear;
	BiParams olayer_linear; // output
	LSTM1Params tweet_left_to_right_lstm_params;
	LSTM1Params tweet_right_to_left_lstm_params;
	LSTM1Params target_left_to_right_lstm_params;
	LSTM1Params target_right_to_left_lstm_params;
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
		olayer_linear.initial(opts.labelSize, opts.inputSize, true, mem);
		tweet_left_to_right_lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
		tweet_right_to_left_lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
		target_left_to_right_lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
		target_right_to_left_lstm_params.initial(opts.hiddenSize, opts.wordDim, mem);
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
		tweet_left_to_right_lstm_params.exportAdaParams(ada);
		tweet_right_to_left_lstm_params.exportAdaParams(ada);
		target_left_to_right_lstm_params.exportAdaParams(ada);
		target_right_to_left_lstm_params.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words E");
		//checkgrad.add(&hidden_linear.W, "hidden w");
		//checkgrad.add(&hidden_linear.b, "hidden b");
		checkgrad.add(&olayer_linear.b, "output layer W");
		checkgrad.add(&olayer_linear.W1, "output layer W");
		checkgrad.add(&olayer_linear.W2, "output layer W");
		checkgrad.add(&tweet_left_to_right_lstm_params.cell.b, "LSTM cell b");
		checkgrad.add(&tweet_left_to_right_lstm_params.cell.W1, "LSTM cell w1");
		checkgrad.add(&tweet_left_to_right_lstm_params.cell.W2, "LSTM cell w2");
		checkgrad.add(&tweet_left_to_right_lstm_params.forget.W1, "LSTM forget w1");
		checkgrad.add(&tweet_left_to_right_lstm_params.forget.W2, "LSTM forget w2");
		checkgrad.add(&tweet_left_to_right_lstm_params.forget.b, "LSTM forget b");
		checkgrad.add(&tweet_left_to_right_lstm_params.input.W1, "LSTM input w1");
		checkgrad.add(&tweet_left_to_right_lstm_params.input.W2, "LSTM input w2");
		checkgrad.add(&tweet_left_to_right_lstm_params.input.b, "LSTM input b");
		checkgrad.add(&tweet_left_to_right_lstm_params.output.W1, "LSTM output w1");
		checkgrad.add(&tweet_left_to_right_lstm_params.output.W2, "LSTM output w2");
		checkgrad.add(&tweet_left_to_right_lstm_params.output.b, "LSTM output b");
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
