Torch implementation of LSTM Language Modelling

- In Terminal.app, run ```python extract_datasets_for_torch.py ``` to generate data for the model. You can adjust vocabsize in the script if you want.
- Run ```th language_model.lua``` to train the model in mini-batch mode. ``` EOSMASK ``` elements are appended to short lines to make them all of equal size in a mini-batch. Mask is applied to prediction and dprediction to zero out influence of the ```EOSMASK``` symbols.