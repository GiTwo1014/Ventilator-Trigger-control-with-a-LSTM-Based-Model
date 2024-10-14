# Ventilator-Trigger-control-with-a-LSTM-Based-Model
In this work, a LSTM-based trigger model was trained to predict the inspiration or expiration events with flow rate. The optimized model is composed of two hidden LSTM layers.


## All data:
```
data-ASL --- All data simulated by the simulator
data-Company --- Data provided by the company
merge.m --- Code for splitting the data
Wavelet.ipynb --- Wavelet transform Time series filtering --- Other filtering methods
```



## file information
```
LSTM model --- LSTM.ipynb
LSTM-FCN model --- LSTM_FCN.ipynb
Time series clustering.ipynb --- Normalize and cluster the segmented window data
Using the step25 dataset:
Transformer model --- transformer.py
```
