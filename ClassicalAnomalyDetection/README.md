# Classical Model

## Using LSTM Auto-Encoder for Anomaly Detection in our Cancer Dataset

### Installation & Usage
1. Make sure you have installed the Python packages in `requirements.txt`
2. Open the jupyter notebook and run all the cells in the notebook
3. The accuracy will be displayed at the bottom of the notebook

### How it Works

We trained our LSTM auto-encoder on the examples of benign tumors so it learns the latent representation of these tumors. 

Then, we plot the losses from the training data and choose a threshold point. Any output loss greater than that threshold point is too different from the benign tumor and will be considered a malignant tumor.
