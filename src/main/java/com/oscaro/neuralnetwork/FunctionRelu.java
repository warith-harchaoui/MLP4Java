package com.oscaro.neuralnetwork;

public class FunctionRelu extends FunctionAbstract {

	public FunctionRelu(String s) throws IllegalArgumentException {
		super(s);
	}

	@Override
	public float[] run(float[] x) {
		float[] res = new float[x.length];
		for (int i = 0; i < res.length; i++) {
			if (x[i] <= 0) {
				res[i] = 0.0f;
			} else {
				res[i] = x[i];
			}
		}
		return res;
	}

	@Override
	public float[] runGradient(float[] intermediateLinearCombination,
			float[] error) {
		float[] res = new float[intermediateLinearCombination.length];
		float tmp;
		for (int i = 0; i < intermediateLinearCombination.length; i++) {
			if (intermediateLinearCombination[i] <= 0) {
				tmp = 0.0f;
			} else {
				tmp = 1.0f;
			}
			res[i] = tmp * error[i];
		}
		return res;
	}

}
