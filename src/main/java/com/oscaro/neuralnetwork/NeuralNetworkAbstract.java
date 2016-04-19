package com.oscaro.neuralnetwork;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

//A NeuralNetwork is an ordered set of Layers (here an Layers[] layers field)
public abstract class NeuralNetworkAbstract {

	protected Layer[] layers; // Main field of a neural network

	public NeuralNetworkAbstract(Layer[] layers)
			throws IllegalArgumentException {
		this.layers = layers;
		if (layers == null || layers.length == 0) {
			throw new IllegalArgumentException("Neural network has no layers");
		}
	}

	public float[] feedForward(float[] x) throws IllegalArgumentException {
		if (layers == null) {
			throw new IllegalArgumentException("The layers should not be null");
		}
		if (layers.length == 0) {
			throw new IllegalArgumentException(
					"The number of layers should not be 0");
		}

		// Recursively feed the network through the layers from bottom (input)
		// to top (output)
		float[] res = layers[0].feedForward(x);// first layer (number 0)
		for (int i = 1; i < layers.length; i++) {
			// pass the result of a layer as an input of the next layer
			res = layers[i].feedForward(res);
		}
		return res;
	}

	protected abstract void backPropagate(float[] target, float[] currentOutput)
			throws IllegalArgumentException;

	public void trainWithOneExample(float[] input, float[] target)
			throws IllegalArgumentException {
		float[] currentOutput = feedForward(input);
		backPropagate(target, currentOutput);
	}

	// TODO: Mini-batch implementation

	@SuppressWarnings("unchecked")
	public JSONObject toJSONObject() {
		JSONObject res = new JSONObject();

		JSONArray layersArray = new JSONArray();
		for (int i = 0; i < layers.length; i++) {
			layersArray.add(layers[i].toJSONObject());
		}

		res.put("layers", layersArray);
		res.put("learningType", this.getClass().getSimpleName());

		return res;
	}

	public static NeuralNetworkAbstract json2NeuralNetwork(JSONObject json)
			throws IllegalArgumentException {
		NeuralNetworkAbstract res = null;

		JSONArray jsonLayers = (JSONArray) json.get("layers");
		Layer[] layers = new Layer[jsonLayers.size()];
		for (int i = 0; i < layers.length; i++) {
			JSONObject jsonLayer = (JSONObject) jsonLayers.get(i);
			layers[i] = Layer.json2Layer(jsonLayer);
		}

		String learningType = (String) json.get("learningType");
		if (learningType.equals("NeuralNetworkCrossEntropyLoss")) {
			res = new NeuralNetworkCrossEntropyLoss(layers);
		} else if (learningType.equals("NeuralNetworkSquareLoss")) {
			res = new NeuralNetworkCrossEntropyLoss(layers);
		} else {
			throw new IllegalArgumentException("Bad JSON, learning type "
					+ learningType + " unknown");
		}

		return res;
	}

}