package com.oscaro.neuralnetwork;

public class FunctionSigmoid extends FunctionAbstract {

	public FunctionSigmoid(String string) throws IllegalArgumentException {
		super(string);
	}

	@Override
	public float[] run(float[] x) {
		float[] res = new float[x.length];
		for (int i = 0; i < res.length; i++) {
			res[i] = (float) (1.0f / (1.0f + Math.exp(-x[i])));
		}
		return res;
	}

	@Override
	public float[] runGradient(float[] intermediateLinearCombination,
			float[] error) {

		float[] tmp = run(intermediateLinearCombination);
		float[] res = new float[tmp.length];

		for (int i = 0; i < tmp.length; i++) {
			res[i] = tmp[i] * (1 - tmp[i]) * error[i];
		}
		return res;
	}

}
