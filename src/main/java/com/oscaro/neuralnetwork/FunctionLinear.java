package com.oscaro.neuralnetwork;

public class FunctionLinear extends FunctionAbstract {

	public FunctionLinear(String s) throws IllegalArgumentException {
		super(s);
	}

	@Override
	public float[] run(float[] x) throws IllegalArgumentException {
		if (x == null || x.length == 0) {
			throw new IllegalArgumentException("Input in layer " + getName()
					+ " should not be null nor empty");
		}
		float[] res = new float[x.length];
		for (int i = 0; i < res.length; i++) {
			res[i] = x[i]; // Identity
		}
		return res;
	}

	@Override
	public float[] runGradient(float[] intermediateLinearCombination,
			float[] error) {

		float[] res = new float[intermediateLinearCombination.length];

		for (int i = 0; i < res.length; i++) {
			res[i] = error[i]; // * 1.0f the derivative of the identity
		}
		return res;
	}

}
