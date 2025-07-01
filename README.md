# 🔊 HyperSound 
> **Author**: *Leonardo Provenzano <leonardoprovenzano21@gmail.com>*  
> **Institution**: *University of Amsterdam*  



##   Data preprocessing

Follow the instruction to download the data set [UrbanSound8K](https://urbansounddataset.weebly.com/) and use the data_preprocessing.ipynb ([mariostrbac])(https://github.com/mariostrbac/environmental-sound-classification.git) to extract the 128x128 mel-spectrogram images



##   Run the models
Run train.py to ouput the labels and store the loss and accuracies.<br>
Change FOLD_K in train.py to train on a differenct cross validation split (do it from 1 to 10 to replicate the results).<br>
Change the model_name in train.py to train a different model.

