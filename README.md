#  Qure by FeeQra / فكرة (Team 1 - NYUAD)


Qure implements a quantum transformer model to forecast time series data such as ECG or EEG and implements a quantum anamoly detector to classify the time series data as normal or anomalous. We compare our results to the classical deep learning counterparts.

 
There are two major components in this project :

* Feature Predictor 
* Anomaly Detector 

The image below shows the workflow of the system.

![alt text](https://github.com/obliviateandsurrender/NYUAD-2023-FeeQra/blob/main/workflow.png)

The [Feature predictor](https://github.com/obliviateandsurrender/NYUAD-2023-FeeQra/blob/main/QRNN.ipynb) predicts the future data and the [Quantum transformer](https://github.com/obliviateandsurrender/NYUAD-2023-FeeQra/blob/main/QuantumSentenceTransformer.py) classifies the data based on the data forecasted by the feature predictor.

References used for the projects : 
> [Transfer learning in hybrid classical-quantum neural networks](https://arxiv.org/abs/1912.08278)

> [Quantum variational rewinding for time series anomaly detection](https://arxiv.org/pdf/2210.16438.pdf)
