package com.oscaro.neuralnetwork;

public class TrainerSGD extends TrainerAbstract {

	private float learningRate;

	public TrainerSGD(String name, float learningRate)
			throws IllegalArgumentException {
		super(name);
		this.learningRate = learningRate;
	}

	public TrainerSGD(String name) throws IllegalArgumentException {
		this(name, 0.01f); // Default value often work but TrainerAdadelta is
							// better
							// than the best value in general
	}

	// Core "algorithm" :)
	@Override
	public void applyGradient(float[][] w, float[][] gradient) {
		for (int i = 0; i < w.length; i++) {
			for (int j = 0; j < w[0].length; j++) {
				w[i][j] = w[i][j] + learningRate * gradient[i][j]; // Hebb rule
			}
		}

	}

}
