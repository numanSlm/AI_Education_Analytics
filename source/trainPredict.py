"""
This script trains a ResNet model on a gender and age dataset using k-fold cross-validation.
It saves the best model for each fold and predicts the labels for a test dataset.
The predictions are saved in a CSV file named "pred_results.csv".
"""

from classes import *
from sklearn.model_selection import KFold, train_test_split
import torch as T

if __name__ == "__main__":
    # Set the paths for the datasets and model
    path_old = "./DataSet_old/"
    path = "./Gender_Age_DataSet/"
    model_path = "./folds/"

    # Set the hyperparameters
    B = 16  # Number of channels in the input image
    N = 8  # Number of residual blocks in the ResNet model
    K = 5  # Number of classes (gender and age categories)
    FC = 2  # Number of fully connected layers in the ResNet model
    AP = 20  # Size of the average pooling layer in the ResNet model
    epochs = 1  # Number of training epochs
    k_folds = 10  # Number of folds for k-fold cross-validation

    # Load the dataset
    ds = ImageDataset(path_old)
    ds_labels = list(ds.classes_dict.keys())

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(ds)):
        train_subsampler = T.utils.data.SubsetRandomSampler(train_idx)
        valid_subsampler = T.utils.data.SubsetRandomSampler(valid_idx)

        # Create train and validation data loaders  with batch size of 64
        train_dl = DataLoader(ds, 64, num_workers=2, pin_memory=True, sampler=train_subsampler)
        val_dl = DataLoader(ds, 64, num_workers=2, pin_memory=True, sampler=valid_subsampler)

        lrs = [0.001]  # Learning rates to try
        wds = [0.0001]  # Weight decays to try

        model_dir = os.path.join(model_path, "B{}_N{}_FC{}_K{}_AP{}_fold{}".format(B, N, FC, K, AP, fold))
        if os.path.isdir(model_dir) == False:
            os.mkdir(model_dir)
        model_name = os.path.join(model_dir, "B{}_N{}_FC{}_K{}_AP{}_fold{}".format(B, N, FC, K, AP, fold))
        model = ResNet(num_classes=4, num_channel=B, num_blocks=N, num_fc_layers=FC, avg_pool_size=AP)
        # The best model will be saved to this path
        save_best_path = os.path.join(model_path, model_name)
        # Load the best model for each fold and predict the labels for the test dataset
        learn = Learner(train_dl, val_dl, model, labels_name=ds_labels, save_best=model_name, log_path=model_name)
        for lr in lrs:
            for wd in wds:
                learn.train_eval(epochs=epochs, lr=lr, wd=wd)

        # Predict labels for the test dataset
        dsp = ImageDataset(path)
        ds_files = dsp.files
        ds_labels = list(dsp.classes_dict.keys())
        test_dl = DataLoader(dsp, 64, num_workers=2, pin_memory=True)
        all_predictions, all_labels, all_probabilites = learn.predict(test_dl)

        # Save the predictions to a CSV file
        df = pd.DataFrame(
            {
                "path": ds_files,
                "true_label": all_labels,
                "pred": all_predictions,
                "prob": all_probabilites,
            }
        )
        df.to_csv("pred_results.csv")
