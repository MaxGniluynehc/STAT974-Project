# STAT974-Project

This repo contains the source code of the STAT974 Project (Fall 2022 semester). 

- [`vol_predictor.py`](https://github.com/MaxGniluynehc/STAT974-Project/blob/main/vol_predictor.py) contains the architecture of the GARCH-LSTM-MLP framework. 
- [`_garch_type_models.py`](https://github.com/MaxGniluynehc/STAT974-Project/blob/main/_garch_type_models.py) contains all the GARCH-type models we considered in the paper. 
- [`dataloader.py`](https://github.com/MaxGniluynehc/STAT974-Project/blob/main/dataloader.py) contains the how we generated batch-samples from the training and test data set, before feeding into the GARCH-LSTM-MLP model. 
- [`_get_data.py`](https://github.com/MaxGniluynehc/STAT974-Project/blob/main/_get_data.py) specifies the data we extracted from [YahooFinance](https://finance.yahoo.com). 
- [`train.py`](https://github.com/MaxGniluynehc/STAT974-Project/blob/main/train.py) and other `.py` files starting with `train` contain the training process and the loss functions, which reflects our modifications and adjustments to the original work. 
- Folders that start with `logs` contain the training logs for different models on different days, as well as the saved models along the training epochs. One can see the model convergence from there. 
