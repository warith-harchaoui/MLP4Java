package com.oscaro.neuralnetwork;

//CrossEntropyLoss is well suited for Classification
public class NeuralNetworkCrossEntropyLoss extends NeuralNetworkAbstract {

	public static float epsThreh = 1e-6f; // for numerical instability code

	public NeuralNetworkCrossEntropyLoss(Layer[] layers)
			throws IllegalArgumentException {
		super(layers);
	}

	@Override
	protected void backPropagate(float[] target, float[] currentOutput)
			throws IllegalArgumentException {
		if (layers == null || layers.length == 0) {
			throw new IllegalArgumentException(
					"The layers should not be null nor empty in neural network");
		}
		if (target == null || target.length == 0) {
			throw new IllegalArgumentException(
					"target should not be null nor empty in neural network");
		}
		if (currentOutput == null || currentOutput.length == 0) {
			throw new IllegalArgumentException(
					"currentOutput should not be null nor empty in neural network");
		}
		if (currentOutput.length != target.length) {
			throw new IllegalArgumentException(
					"currentOutput and target should have the same size in neural network");
		}

		float[] error = new float[currentOutput.length];
		for (int i = 0; i < error.length; i++) {
			error[i] = target[i] / (currentOutput[i]); // Derivative of the
														// cross-entropy loss

			// Handling numerical instability
			if (target[i] < epsThreh || currentOutput[i] < epsThreh) {

				if (currentOutput[i] > epsThreh) {// && target[i] < epsThreh
					error[i] = 0.0f;
				} else {
					if (Float.isInfinite(error[i])){
						error[i] = 1.0f;
					} else if(Float.isNaN(error[i])){
						error[i] = 1.0f;
					}
				}

			}
		}

		// Recursively propagate the error through the neural network from top
		// (output) to bottom (input)
		error = layers[layers.length - 1].backPropagate(error);

		for (int i = layers.length - 2; i >= 0; i--) {
			error = layers[i].backPropagate(error);
		}
	}
}
