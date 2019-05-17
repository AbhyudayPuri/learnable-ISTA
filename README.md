# learnable-ISTA
This is an implementation of the Learnable Iterative Shrinkage and Thresholding Algorithm for the purpose of image denoising. 

#### To walk through the results of this paper, please refer to the main.ipynb notebook.

## To learn the dictionary:
```bash
python main.py
```
Before training the encoder network, you must create and organize the data for the same. To do so, run the following commands:
```bash
cd utils 
python create_train_data.py
python preprocess_data.py 
```

## To train the encoder network
```bash
python train.py
```

Incase you face any errors in running the above two commands, just check for the paths, as absolute paths have been given, instead of relative. So this might change depending on your system, or file organization. 