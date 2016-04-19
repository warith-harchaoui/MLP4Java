package com.oscaro.neuralnetwork;

public abstract class TrainerAbstract {

	protected String name;

	public String getName() {
		return name;
	}

	public TrainerAbstract(String name) throws IllegalArgumentException {
		if(name==null || name.trim().length()==0){
			throw new IllegalArgumentException("Name of trainer should not be null nor empty nor spaces");
		}
		this.name = name;
	}

	// The difference between trainers is the applyGradient function only
	public abstract void applyGradient(float[][] w, float[][] gradient) throws IllegalArgumentException;

}
