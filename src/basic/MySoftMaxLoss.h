#ifndef MY_SOFTMAXLOSS_H_
#define MY_SOFTMAXLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include "Node.h"
#include "Stance.h"


class MySoftMaxLoss{
public:
	inline dtype loss(PNode x, const vector<dtype> &answer, Metric& favorMetric, Metric &againstMetric, Metric &neuralMetric, int batchsize = 1){
		int nDim = x->dim;
		int labelsize = answer.size();
		if (labelsize != nDim) {
			std::cerr << "softmax_loss error: dim size invalid" << std::endl;
			return -1.0;
		}

		NRVec<dtype> scores(nDim);

		dtype cost = 0.0;
		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			if (answer[i] >= 0) {
				if (optLabel < 0 || x->val[i] > x->val[optLabel])
					optLabel = i;
			}
		}

		dtype sum1 = 0, sum2 = 0, maxScore = x->val[optLabel];
		for (int i = 0; i < nDim; ++i) {
			scores[i] = -1e10;
			if (answer[i] >= 0) {
				scores[i] = exp(x->val[i] - maxScore);
				if (isEqual(answer[i] , 1))
					sum1 += scores[i];
				sum2 += scores[i];
			}
		}
		cost += (log(sum2) - log(sum1)) / batchsize;
		if (optLabel == Stance::FAVOR) {
			if (isEqual(answer[optLabel], 1))
				favorMetric.correct_label_count++;
			favorMetric.predicated_label_count++;
		}
		if (isEqual(answer[Stance::FAVOR], 1)) {
			favorMetric.overall_label_count++;
		}
		if (isEqual(optLabel, Stance::AGAINST)) {
			if (isEqual(answer[optLabel] , 1))
				againstMetric.correct_label_count++;
			againstMetric.predicated_label_count++;
		}
		if (isEqual(answer[Stance::AGAINST] , 1)) {
			againstMetric.overall_label_count++;
		}
		if (isEqual(optLabel, Stance::NONE)) {
			if (isEqual(answer[optLabel] , 1))
				neuralMetric.correct_label_count++;
			neuralMetric.predicated_label_count++;
		}
		if (isEqual(answer[Stance::NONE] , 1)) {
			neuralMetric.overall_label_count++;
		}

		for (int i = 0; i < nDim; ++i) {
			if (answer[i] >= 0) {
				x->loss[i] = (scores[i] / sum2 - answer[i]) / batchsize;
			}
		}
		
		return cost;
	}

	inline dtype predict(PNode x, int& y){
		int nDim = x->dim;

		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			if (optLabel < 0 || x->val[i] >  x->val[optLabel])
				optLabel = i;
		}

		dtype prob = 0.0;
		dtype sum = 0.0;
		NRVec<dtype> scores(nDim);
		dtype maxScore = x->val[optLabel];
		for (int i = 0; i < nDim; ++i) {
			scores[i] = exp(x->val[i] - maxScore);
			sum += scores[i];
		}
		prob = scores[optLabel] / sum;
		y = optLabel;
		return prob;
	}

	inline dtype cost(PNode x, const vector<dtype> &answer, int batchsize = 1){
		int nDim = x->dim;
		int labelsize = answer.size();
		if (labelsize != nDim) {
			std::cerr << "softmax_loss error: dim size invalid" << std::endl;
			return -1.0;
		}

		NRVec<dtype> scores(nDim);

		dtype cost = 0.0;

		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			if (answer[i] >= 0) {
				if (optLabel < 0 || x->val[i] > x->val[optLabel])
					optLabel = i;
			}
		}

		dtype sum1 = 0, sum2 = 0, maxScore = x->val[optLabel];
		for (int i = 0; i < nDim; ++i) {
			scores[i] = -1e10;
			if (answer[i] >= 0) {
				scores[i] = exp(x->val[i] - maxScore);
				if (isEqual(answer[i],  1))
					sum1 += scores[i];
				sum2 += scores[i];
			}
		}
		cost += (log(sum2) - log(sum1)) / batchsize;
		return cost;
	}

};


#endif /* _SOFTMAXLOSS_H_ */
