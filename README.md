# Persuasion-Advertisements

This repository is the implementation of the paper :
> Yaman Kumar Singla, Rajat Jha, Aditya Garg, Ayush Bhardwaj, Tushar, Arunim Gupta, Milan Aggarwal, Balaji Krishnamurthy, Diyi Yang, Rajiv Ratn Shah, and Changyou Chen. "Persuasion Strategies in Advertisements" (2022).

### Model Architecture
![image](ADVISE-CODE/model/Persuasion Arch Diag.png "Model Architecture Diagram")

### Requirements
Use the `environment.yml` file to set up the `conda` environment.
```
$ conda env create -n ENVNAME --file environment.yml
```
### Training
To train the model, run the following command:
```
$ python train.py --gpu_id <gpu id> --epochs <number of epochs for training> --model_name <name for trained model>
```
