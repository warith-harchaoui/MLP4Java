package com.oscaro.neuralnetwork;

import java.util.Random;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class Layer {

	protected float[][] w;// weights

	protected float[] input; // inputs and outputs are stored for
								// Back-Propagation

	protected float[] intermediateLinearCombination;// just before the
													// activation function
	protected float[] output;

	protected FunctionAbstract f;

	// One trainer by layer because for example in SGD, the learning rate is
	// defined for each layer -- counter-intuitive as a trainer should be on top
	// of the layer normally but I found out it is the best this way
	protected TrainerAbstract trainer;

	public String getName() {
		return f.getName();// Name is here for debugging = function f's name
	}

	private Layer(float[][] w, FunctionAbstract f)
			throws IllegalArgumentException {
		this.w = w;
		this.f = f;
		if (f == null) {
			throw new IllegalArgumentException(
					"ActivationFunction f of Neural Network should not be null in layer ");
		}
		if (w == null || w.length == 0 || w[0].length == 0) {
			throw new IllegalArgumentException(
					"Matrix w of Neural Network should not be null nor empty in layer "
							+ getName());
		}
	}

	private Layer(float[][] w, FunctionAbstract f, TrainerAbstract trainer)
			throws IllegalArgumentException {
		this(w, f);
		if (trainer == null) {
			throw new IllegalArgumentException(
					"Trainer should not be null in layer " + getName());
		}
		this.trainer = trainer;
	}

	public Layer(int inputSize, int outputSize, FunctionAbstract f,
			TrainerAbstract trainer) throws IllegalArgumentException {
		this(new float[outputSize][inputSize + 1], f, trainer);// +1 for bias
																// (in dot
																// product +
																// bias)
		Random random = new Random();
		for (int i = 0; i < w.length; i++) {
			for (int j = 0; j < w[0].length - 1; j++) {
				// Xavier's initialization in
				// Xavier Glorot & Yoshua Bengio’s
				// "Understanding the difficulty of training deep feedforward neural networks"
				w[i][j] = (float) ((Math.sqrt(6.0 / (outputSize + inputSize))) * 2.0f * (random
						.nextFloat() - 0.5));
			}
			w[i][w[0].length - 1] = 0.0f;// bias is set to 0
		}
	}

	// Outputs the result of the net for a given x as input
	public float[] feedForward(float[] x) throws IllegalArgumentException {
		if (x == null || x.length == 0) {
			throw new IllegalArgumentException(
					"x should not be null nor empty in Layer " + getName());
		}

		input = new float[x.length + 1];
		for (int i = 0; i < x.length; i++) {
			input[i] = x[i];
		}
		input[x.length] = 1.0f;

		intermediateLinearCombination = dotProduct(w, input);

		// Activation function
		output = f.run(intermediateLinearCombination);

		// Deep copy of output
		float[] res = new float[output.length];
		for (int i = 0; i < output.length; i++) {
			res[i] = output[i];
		}

		return res;
	}

	protected float[] dotProduct(float[][] A, float[] b)
			throws IllegalArgumentException {
		float[] res = new float[A.length];
		if (A == null || b == null || A.length == 0 || b.length == 0) {
			throw new IllegalArgumentException(
					"dotProduct should have descent dimensions (different from 0 or null) in layer "
							+ getName());
		}
		if (A[0].length != b.length) {
			throw new IllegalArgumentException("Wrong dot product in layer "
					+ getName());
		}
		for (int i = 0; i < A.length; i++) {
			// Linear combination
			for (int j = 0; j < A[0].length; j++) {
				res[i] += A[i][j] * b[j];
			}
		}
		return res;
	}

	public float[] backPropagate(float[] error) throws IllegalArgumentException {
		if (error == null || error.length == 0) {
			throw new IllegalArgumentException(
					"backPropagate have invalid error vector in layer "
							+ getName());
		}

		float[] nextError = new float[input.length];

		float[] d = f.runGradient(intermediateLinearCombination, error);
		// System.out.println(printVector(d));

		// Gradient
		float[][] g = new float[output.length][input.length];
		for (int i = 0; i < output.length; i++) {
			for (int j = 0; j < input.length; j++) {
				nextError[j] += w[i][j] * d[i];
				g[i][j] = input[j] * d[i];
			}
		}

		// Side effect that modifies w
		trainer.applyGradient(w, g);

		return nextError;
	}

	@SuppressWarnings("unchecked")
	public JSONObject toJSONObject() {
		JSONObject res = new JSONObject();

		JSONArray wArray = new JSONArray();
		for (int i = 0; i < w.length; i++) {
			JSONArray wRow = new JSONArray();
			for (int j = 0; j < w[0].length; j++) {
				wRow.add(new Double(w[i][j]));
			}
			wArray.add(wRow);
		}

		res.put("functionType", f.getClass().getSimpleName());
		res.put("name", getName());
		res.put("w", wArray);

		return res;
	}

	public static Layer json2Layer(JSONObject json)
			throws IllegalArgumentException {

		JSONArray jsonW = (JSONArray) json.get("w");
		JSONArray jsonWRow = (JSONArray) jsonW.get(0);
		float[][] w = new float[jsonW.size()][jsonWRow.size()];
		for (int i = 0; i < w.length; i++) {
			jsonWRow = (JSONArray) jsonW.get(i);
			for (int j = 0; j < w[0].length; j++) {
				w[i][j] = ((Double) jsonWRow.get(j)).floatValue();
			}
		}

		String functionType = (String) json.get("functionType");
		FunctionAbstract f = null;
		String name = (String) json.get("name");
		if (functionType.equals("FunctionLinear")) {
			f = new FunctionLinear(name);
		} else if (functionType.equals("FunctionRelu")) {
			f = new FunctionRelu(name);
		} else if (functionType.equals("FunctionSigmoid")) {
			f = new FunctionSigmoid(name);
		} else if (functionType.equals("FunctionSoftmax")) {
			f = new FunctionSoftmax(name);
		} else {
			throw new IllegalArgumentException("Bad JSON, function type "
					+ functionType + " unknown");
		}

		return new Layer(w, f);
	}
}
