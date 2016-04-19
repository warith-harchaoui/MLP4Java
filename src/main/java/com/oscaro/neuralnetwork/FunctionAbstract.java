package com.oscaro.neuralnetwork;

public abstract class FunctionAbstract {

	// Names are useful for debugging to know exactly where it gives...
	protected String name;

	public String getName() {
		return name;
	}

	public FunctionAbstract(String name) throws IllegalArgumentException {
		if (name == null || name.trim().length() == 0) {
			throw new IllegalArgumentException(
					"Name of function should not be null not empty not spaces only");
		}
		this.name = name;
	}

	// Computation of function
	public abstract float[] run(float[] x) throws IllegalArgumentException;

	// Computation of its derivative
	public abstract float[] runGradient(float[] intermediateLinearCombination,
			float[] error) throws IllegalArgumentException;
}
