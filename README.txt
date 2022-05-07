# AD-LSTMVAE_fire_detection

This is the repo for along short-term memory variational autoencoder (LSTM-VAE) code.

It includes:

(1) main.py: main file serves as an example to train, test, reload, and save the model;\
(2) ad_lstmvae.py: class includes function of initializing, training, optimizing, testing, loading, and saving LSTMVAE; \
(3) LSTMModel.py: class include the encoder and decoder models;\
(4) utility.py: utility functions for data loading, signal reshaping, Kalman fitering, and fire alarm criteria;\
(5) exampleData.csv: an example data for periodic change detection;\
(6) AD_LSTM_example: example output: timestep: Current time; value: signal value; anomaly_probability: the probability of anomaly;\
(7) model.pt: a pretrained model for the example dataset anomaly detection;\

Note: Code written and conducted on a Anaconda (v1.10.0) platfrom with Python 3.7.3 on a Windows 10 machine

Code written and conducted on a Anaconda (v1.10.0) platfrom with Python 3.7.3 on a Windows 10 machine

Extra packages:

(1) Pytorch: v1.7.0

(2) pykalman v0.9.2: https://pykalman.github.io/
