package com.oscaro.neuralnetwork;

import org.json.simple.JSONObject;
import org.junit.Test;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;

public class SanityChecksTests {

	protected static final float epsilon = 1e-1f;// Numerical precision

	public Object[] xorDataset() {
		List<float[]> data = new ArrayList<float[]>();
		List<float[]> labels = new ArrayList<float[]>();

		// XOR dataset
		data.add(new float[] { 1, 0 });
		labels.add(new float[] { 1, 0 });// For binary classification, a float
											// of length 1 would be enough but
											// does generalize to K>1 classes

		data.add(new float[] { 0, 1 });
		labels.add(new float[] { 1, 0 });

		data.add(new float[] { 1, 1 });
		labels.add(new float[] { 0, 1 });

		data.add(new float[] { 0, 0 });
		labels.add(new float[] { 0, 1 });

		Object[] res = new Object[2];
		res[0] = data;
		res[1] = labels;
		return res;
	}

	@Test
	public void testSGDSoftmaxCrossEntropyLoss()
			throws IllegalArgumentException {

		Object[] dataset = xorDataset();
		@SuppressWarnings("unchecked")
		List<float[]> data = (List<float[]>) dataset[0];
		@SuppressWarnings("unchecked")
		List<float[]> labels = (List<float[]>) dataset[1];

		int dimInput = data.get(0).length;
		int dimOutput = labels.get(0).length;
		int nbEpochs = 100000;

		int[] dimsLayers = { 3, dimOutput };

		// Building the network's structure
		int prev = dimInput;
		Layer[] layers = new Layer[dimsLayers.length];
		for (int i = 0; i < dimsLayers.length; i++) {
			FunctionAbstract f = null;
			String s = null;
			if (i == dimsLayers.length - 1) {
				s = "Softmax last level " + (dimsLayers.length - 1);
				// Last layer is a soft max (better for exclusive classes)
				f = new FunctionSoftmax(s);
			} else {
				s = "Sigmoid level " + i;
				f = new FunctionSigmoid(s);
			}
			layers[i] = new Layer(prev, dimsLayers[i], f, new TrainerSGD(
					f.getName()));

			prev = dimsLayers[i];
		}
		NeuralNetworkAbstract nnet = new NeuralNetworkCrossEntropyLoss(layers);

		// Training the network
		for (int iter = 0; iter < nbEpochs; iter++) {
			for (int i = 0; i < data.size(); i++) {
				nnet.trainWithOneExample(data.get(i), labels.get(i));
			}
		}

		float err;
		for (int i = 0; i < data.size(); i++) {
			float[] output = nnet.feedForward(data.get(i));
			float[] trueLabel = labels.get(i);
			for (int j = 0; j < output.length; j++) {
				err = Math.abs(output[j] - trueLabel[j]);
				assertTrue(
						"SGD Softmax with CrossEntropyLoss Test did not converge correctly, error is too big: "
								+ err, err < epsilon);
			}
		}

	}

	@Test
	public void testSGDSigmoidSquareLoss() throws IllegalArgumentException {

		Object[] dataset = xorDataset();
		@SuppressWarnings("unchecked")
		List<float[]> data = (List<float[]>) dataset[0];
		@SuppressWarnings("unchecked")
		List<float[]> labels = (List<float[]>) dataset[1];

		int dimInput = data.get(0).length;
		int dimOutput = labels.get(0).length;
		int nbEpochs = 100000;

		int[] dimsLayers = { 3, dimOutput };

		// Building the network's structure
		int prev = dimInput;
		Layer[] layers = new Layer[dimsLayers.length];
		for (int i = 0; i < dimsLayers.length; i++) {
			FunctionAbstract f = null;
			String s = null;
			if (i == dimsLayers.length - 1) {
				s = "Sigmoid last level " + (dimsLayers.length - 1);
				// Last layer is a sigmoid (well-suited for non-exclusive
				// classes)
				f = new FunctionSigmoid(s);
			} else {
				s = "Sigmoid level " + i;
				f = new FunctionSigmoid(s);
			}
			layers[i] = new Layer(prev, dimsLayers[i], f, new TrainerSGD(
					f.getName()));

			prev = dimsLayers[i];
		}
		NeuralNetworkAbstract nnet = new NeuralNetworkSquareLoss(layers);

		// Training the network
		for (int iter = 0; iter < nbEpochs; iter++) {
			for (int i = 0; i < data.size(); i++) {
				nnet.trainWithOneExample(data.get(i), labels.get(i));
			}
		}

		float err;
		for (int i = 0; i < data.size(); i++) {
			float[] output = nnet.feedForward(data.get(i));
			float[] trueLabel = labels.get(i);

			for (int j = 0; j < output.length; j++) {
				err = Math.abs(output[j] - trueLabel[j]);
				assertTrue(
						"SGD Sigmoid with Square Loss Test did not converge correctly, error is too big: "
								+ err, err < epsilon);
			}
		}

	}

	@Test
	public void testAdadeltaSoftmaxCrossEntropyLoss()
			throws IllegalArgumentException {

		Object[] dataset = xorDataset();
		@SuppressWarnings("unchecked")
		List<float[]> data = (List<float[]>) dataset[0];
		@SuppressWarnings("unchecked")
		List<float[]> labels = (List<float[]>) dataset[1];

		int dimInput = data.get(0).length;
		int dimOutput = labels.get(0).length;
		int nbEpochs = 100000;

		int[] dimsLayers = { 3, dimOutput };

		// Building the network's structure
		int prev = dimInput;
		Layer[] layers = new Layer[dimsLayers.length];
		for (int i = 0; i < dimsLayers.length; i++) {
			FunctionAbstract f = null;
			String s = null;
			if (i == dimsLayers.length - 1) {
				s = "Softmax last level " + (dimsLayers.length - 1);
				// Last layer is a soft max (better for exclusive classes)
				f = new FunctionSoftmax(s);
			} else {
				s = "Sigmoid level " + i;
				f = new FunctionSigmoid(s);
			}
			layers[i] = new Layer(prev, dimsLayers[i], f, new TrainerAdadelta(
					f.getName()));

			prev = dimsLayers[i];
		}
		NeuralNetworkAbstract nnet = new NeuralNetworkCrossEntropyLoss(layers);

		// Training the network
		for (int iter = 0; iter < nbEpochs; iter++) {
			for (int i = 0; i < data.size(); i++) {
				nnet.trainWithOneExample(data.get(i), labels.get(i));
			}
		}

		float err;
		for (int i = 0; i < data.size(); i++) {
			float[] output = nnet.feedForward(data.get(i));
			float[] trueLabel = labels.get(i);
			for (int j = 0; j < output.length; j++) {
				err = Math.abs(output[j] - trueLabel[j]);
				assertTrue(
						"Adadelta Softmax with CrossEntropy Loss Test did not converge correctly, error is too big: "
								+ err, err < epsilon);
			}
		}

	}

	@Test
	public void testJSON() throws IllegalArgumentException {
		Object[] dataset = xorDataset();
		@SuppressWarnings("unchecked")
		List<float[]> data = (List<float[]>) dataset[0];
		@SuppressWarnings("unchecked")
		List<float[]> labels = (List<float[]>) dataset[1];

		int dimInput = data.get(0).length;
		int dimOutput = labels.get(0).length;

		int[] dimsLayers = { 3, dimOutput };

		// Building the network's structure
		int prev = dimInput;
		Layer[] layers = new Layer[dimsLayers.length];
		for (int i = 0; i < dimsLayers.length; i++) {
			FunctionAbstract f = null;
			String s = null;
			if (i == dimsLayers.length - 1) {
				s = "Softmax last level " + (dimsLayers.length - 1);
				// Last layer is a soft max (better for exclusive classes)
				f = new FunctionSoftmax(s);
			} else {
				s = "Sigmoid level " + i;
				f = new FunctionSigmoid(s);
			}
			layers[i] = new Layer(prev, dimsLayers[i], f, new TrainerAdadelta(
					f.getName()));

			prev = dimsLayers[i];
		}
		NeuralNetworkAbstract nnet = new NeuralNetworkCrossEntropyLoss(layers);

		// No need of training because the weights are randomly set in layers

		// Serialization
		JSONObject jsonNnet = nnet.toJSONObject();

		// De-serialization
		NeuralNetworkAbstract nnet2 = NeuralNetworkAbstract
				.json2NeuralNetwork(jsonNnet);

		float err;
		for (int i = 0; i < data.size(); i++) {
			float[] output = nnet.feedForward(data.get(i));// Output from first nnet
			float[] output2 = nnet2.feedForward(data.get(i));// Output from second nnet
			for (int j = 0; j < output.length; j++) {
				err = Math.abs(output[j] - output2[j]);
				assertTrue(
						"Output from original nnet and the one JSON-imported are different, eps: "
								+ err, err < epsilon);
			}
		}

	}

}
