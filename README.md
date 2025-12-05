
# 02456DeepLearning_Project3_Grp51
Project 3 - Reliable uncertainty estimation for neural networks with conformal prediction

ABSTRACT

This project was completed as part of our Deep Learning module at the Technical University of Denmark for our respective exchange programs. 

By Group 51:
- My Lan Nguyen (BASc Biomedical Engineering, UBC)
- Saffron Salmah Yen Lim (BComp Data Science & Artificial Intelligence, NTU Singapore)
- Fatima Nowshad (BComp Data Science & Artificial Intelligence, NTU Singapore)
- Thorri Elis Halldoruson

# File Directories
Please refer to code/Report_code.ipynb for the main results.
```
code/
├── data/
│ └── cifar-10-batches-py/
│ ├── batches.meta
│ ├── data_batch_1
│ ├── data_batch_2
│ ├── data_batch_3
│ ├── data_batch_4
│ ├── data_batch_5
│ ├── readme.html
│ └── test_batch
|
├── examples/
│ └── scratch/
│ └── rhti/
│ └── conformal/
│ ├── data/
│ │ ├── cifar10_resnet18.pth
│ │ ├── holdout_features.npz
│ │ ├── holdout_predictions.pth
│ │ ├── test_predictions.pth
│ │ ├── val_features.npz
│ │ └── val_predictions.pth
| 
├── Report_code.ipynb ← Contains runnable code & relevant results
├── init.py
├── cluster_conformal.py
├── conformal_test.py
├── data.py
├── evaluate_test.py
├── extract_features.py
├── knn_conformal.py
├── train.py
├── train_cifar10.py
.gitignore
README.md
requirements.txt
setup.py
```

