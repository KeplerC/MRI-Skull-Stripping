## Running these models 



### Baseline models 

the baseline models are written in jupyternotebook. Because of the limitation of hardware, I cannot run the graphs(like learning curves for 80,000 pixels) on a laptop computer, so that notebook is just a proof-of-concept and I used only two images to do the whole thing.

All those codes were later exported to python scripts and run on a Amazon VPS. 

However, code that I used to generate those graphs on the report are the same, except what's inside ./Baseline/data is different. 



### CNN

For CNN models, those codes, which are based on **Tensorflow**, can generalize to a lot of models in ./CNN/autoencoder.py. 

./CNN/config.py are hyperparameters and ./CNN/dataset_handler is what we feed from dcm to batches . 

One can modify ./CNN/run to run the whole code.