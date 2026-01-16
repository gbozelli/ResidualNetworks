# Residual Neural Network for Signal Compensation in Optical Communications

## Overview
This project evaluates the performance of Machine Learning (ML) architectures—specifically Multi-Layer Perceptrons (MLP) and 1D Convolutional Neural Networks (WIP) for the mitigation of non-linear impairments in optical fiber systems. The study focuses on a comparative analysis between a **Residual Learning (ResNet)** framework and a **Direct Mapping (Normal)** framework across various Signal-to-Noise Ratio (SNR) scenarios, measured in dBm.

## Methodology

### 1. Data Processing and Windowing
The input signal consists of four channels ($XI, XQ, YI, YQ$) representing dual-polarization in-phase and quadrature components. A sliding window mechanism of size $n_{sym}$ is implemented to capture temporal dependencies and inter-symbol interference (ISI).

* **Direct Mapping:** The model is trained to map the noisy received sequence directly to the transmitted symbols.
* **Residual Mapping (ResNet):** The model is trained to predict the residual noise (the difference between the transmitted symbol and the center of the received window). The final estimation is obtained by $Y_{hat} = X_{center} + Y_{Residualpredicted}$.

### 2. Experimental Scenarios
The system performs an automated grid search across the following hyperparameter space:
* **Launch Power (dBm):** [9, 10, 11]
* **Network Complexity (Neurons):** [25, 50, 75, 100]
* **Memory Depth (Symbol Window):** [1, 3, 5, 7]

### 3. Model Architectures

#### Multi-Layer Perceptron (MLP)
Implemented via `sklearn.neural_network.MLPRegressor`, utilizing `partial_fit` for incremental learning. This model processes flattened feature vectors.
* **Input Layer:** $(n_{sym} \times 4)$
* **Flattening & Fully Connected:** Variable neurons, ReLU activation.
* **Output Layer:** 4 linear units (regressing $XI, XQ, YI, YQ$).

## Performance Evaluation
The primary metric is the **Bit Error Ratio (BER)**. The project calculates the BER at each training epoch to monitor convergence.
* Symbols are decoded into 16-QAM constellations.
* BER is derived from the Symbol Error Ratio (SER) assuming Gray coding.
* Results are exported as high-resolution PDF plots for scientific reporting.

## Dependencies
* Python 3.x
* NumPy
* Scikit-learn
* TensorFlow / Keras (WIP)
* Matplotlib

## Usage
Execute the main script to initiate the grid search. The results will be stored in the root directory as `CNN_DBM{P}_nsym{N}_neu{M}.pdf`.
