"""
This script performs retraining of a ResNet model using k-fold cross-validation.
It loads an image dataset, splits it into training and validation sets using k-fold cross-validation,
trains the model on the training set, evaluates it on the validation set, and saves the best model.
Finally, it predicts the labels for a test dataset and saves the predictions to a CSV file.
"""

from classes import *
from sklearn.model_selection import KFold, train_test_split
import torch as T

# Main function guard
if __name__ == "__main__":
    path = "./Data_retrain/"
    model_path = "./retrain_model/"
    B = 16
    N = 8
    K = 5
    FC = 2
    AP = 20
    epochs = 5
    k_folds = 10

    ds = ImageDataset(path)
    df = pd.read_csv("biased.csv")
    ds.files = list(df["path"])
    print(ds.files)
    # labels
    ds_labels = list(ds.classes_dict.keys())

    # Kfold split 
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # create train data loaders for each set with batch size of 64
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(ds)):
        if fold == 8:
            train_subsampler = T.utils.data.SubsetRandomSampler(train_idx)
            valid_subsampler = T.utils.data.SubsetRandomSampler(valid_idx)

            # ceate a validation and train set with batch size of 64
            train_dl = DataLoader(ds, 64, num_workers=2, pin_memory=True, sampler=train_subsampler)
            val_dl = DataLoader(ds, 64, num_workers=2, pin_memory=True, sampler=valid_subsampler)

            # block size
            lrs = [0.001]  # Learning rates
            wds = [0.0001]  # Weight decays

            # create model and learner objects and train the model on the training set and evaluate it on the validation set
            model_dir = os.path.join(model_path, "B{}_N{}_FC{}_K{}_AP{}_fold{}".format(B, N, FC, K, AP, fold))
            if os.path.isdir(model_dir) == False:
                os.mkdir(model_dir)
            model_name = os.path.join(model_dir, "B{}_N{}_FC{}_K{}_AP{}_fold{}".format(B, N, FC, K, AP, fold))
            model = ResNet(num_classes=4, num_channel=B, num_blocks=N, num_fc_layers=FC, avg_pool_size=AP)

            # save best model path and name 
            save_best_path = os.path.join(model_path, model_name)

            # load the best model
            learn = Learner(train_dl, val_dl, model, labels_name=ds_labels, save_best=model_name, log_path=model_name)
            learn.load("/home/dara/Professional/programming/concordia_AI_project/Part3/best_model/B16_N8_FC2_K5_AP20_fold8")
            for lr in lrs:
                for wd in wds:
                    learn.train_eval(epochs=epochs, lr=lr, wd=wd)
            # learn.plot_metrics_macro(model_name + "metrics_mocro.svg")
            dsp = ImageDataset(path)
            ds_files = dsp.files

            # labels 
            ds_labels = list(dsp.classes_dict.keys())
            # create test data loader with batch size of 64
            test_dl = DataLoader(dsp, 64, num_workers=2, pin_memory=True)
            # predict the labels for the test set
            all_predictions, all_labels, all_probabilites = learn.predict(test_dl)

            df = pd.DataFrame(
                {
                    "path": ds_files,
                    "true_label": all_labels,
                    "pred": all_predictions,
                    "prob": all_probabilites,
                }
            )
            df.to_csv("pred_resutlsRe.csv")
