package com.oscaro.neuralnetwork;

import java.util.Arrays;

public class FunctionSoftmax extends FunctionAbstract {

	public FunctionSoftmax(String string) throws IllegalArgumentException {
		super(string);
	}

	@Override
	public float[] run(float[] x) {
		float[] res = new float[x.length];

		// ValMax is here for numerical stability of the Math.exp function
		float valMax = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < x.length; i++) {
			if (x[i] > valMax) {
				valMax = x[i];
			}
		}

		float z = 0.0f;
		// Exponentiation
		for (int i = 0; i < x.length; i++) {
			// Dividing both numerator and denominator by Math.exp(-valMax) does
			// not change the result but avoid numerical instability
			res[i] = (float) (Math.exp(x[i] - valMax));
			z += res[i];
		}
		// Normalization where w is the denominator
		for (int i = 0; i < x.length; i++) {
			res[i] = res[i] / z;
		}
		return res;
	}

	@Override
	public float[] runGradient(float[] intermediateLinearCombination,
			float[] error)  {

		float[] tmp = run(intermediateLinearCombination);

		float[] res = new float[tmp.length];
		Arrays.fill(res, 0.0f);

		float delta_ij;
		for (int i = 0; i < tmp.length; i++) {
			for (int j = 0; j < tmp.length; j++) {
				delta_ij = 0.0f;
				if (i == j) {
					delta_ij = 1.0f;
				}
				res[i] += (tmp[i]) * (delta_ij - (tmp[j])) * error[j];
			}
		}
		return res;
	}
}
