package com.oscaro.neuralnetwork;

import java.util.Arrays;

// Located at each layer
public class TrainerAdadelta extends TrainerAbstract {

	// Notations and algorithm from original Adadelta paper by Zeiler (Algorithm
	// 1 page 3)
	private float rho;
	private float epsilon;

	// Memory state
	private float[][] eG2;// Gradient Accumulator
	private float[][] eDeltaX2;// Delta Accumulator

	public TrainerAdadelta(String name, float rho, float epsilon) throws IllegalArgumentException {
		super(name);
		if(rho<0 || epsilon<0){
			throw new IllegalArgumentException("Both rho and epsilon should be non-negative in layer "+getName());
		}
		if(epsilon==0){
			throw new IllegalArgumentException("epsilon=0 will create numerical instability at convergence in layer "+getName());			
		}
		this.rho = rho;
		this.epsilon = epsilon;
		eG2 = null;
		eDeltaX2 = null;
	}

	public TrainerAdadelta(String name) throws IllegalArgumentException {
		// Default values that always work :) (major advantage of Adadelta: no
		// learning rates to tune)
		this(name, 0.95f, 1e-6f);
	}

	// Core algorithm
	@Override
	public void applyGradient(float[][] w, float[][] gradient) throws IllegalArgumentException {
		if(w==null || w.length==0 || w[0].length==0){
			throw new IllegalArgumentException("w should not be null nor zero in trainer Adadelta in layer "+getName());
		}

		// Lazy initialization of both eG2 and eDeltaX2
		if (eG2 == null) {
			eG2 = new float[w.length][w[0].length];
			eDeltaX2 = new float[w.length][w[0].length];
			for (int i = 0; i < eG2.length; i++) {
				Arrays.fill(eG2[i], 0);
				Arrays.fill(eDeltaX2[i], 0);
			}
		}

		// Accumulate Gradient
		for (int i = 0; i < eG2.length; i++) {
			for (int j = 0; j < eG2[0].length; j++) {
				eG2[i][j] = rho * eG2[i][j] + (1.0f - rho) * gradient[i][j]
						* gradient[i][j];
			}
		}

		// Compute update
		float[][] delta = new float[w.length][w[0].length];
		for (int i = 0; i < delta.length; i++) {
			for (int j = 0; j < delta[0].length; j++) {
				delta[i][j] = RMS(eDeltaX2[i][j]) / RMS(eG2[i][j])
						* gradient[i][j];
			}
		}

		// Accumulate Updates
		for (int i = 0; i < eDeltaX2.length; i++) {
			for (int j = 0; j < eDeltaX2[0].length; j++) {
				eDeltaX2[i][j] = rho * eDeltaX2[i][j] + (1.0f - rho)
						* delta[i][j] * delta[i][j];
			}
		}

		for (int i = 0; i < w.length; i++) {
			for (int j = 0; j < w[0].length; j++) {
				w[i][j] = w[i][j] + delta[i][j];
			}
		}

	}

	// Also defined in paper
	public float RMS(float x) {
		return (float) Math.sqrt(epsilon + x);
	}

}
