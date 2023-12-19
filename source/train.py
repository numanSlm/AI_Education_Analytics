from classes import *

# Main function guard
if __name__ == "__main__":
    path = "./DataSet/"
    model_path ="./models/"
    ds = ImageDataset(path)

    # labels
    ds_labels = list(ds.classes_dict.keys())
    # split the dataset to train, validation and test sets with 70%, 15%, 15% respectively
    train_set, val_set, test_set = T.utils.data.random_split(ds, [0.7,0.15,0.15])

    # create train data loaders for each set with batch size of 64
    train_dl = DataLoader(train_set, 64, shuffle=True, num_workers=2, pin_memory=True)
    # ceate a validation set with batch size of 64
    val_dl = DataLoader(val_set, 64, shuffle=True, num_workers=2, pin_memory=True)
    # ceate a test set with batch size of 64
    test_dl = DataLoader(test_set, 64, shuffle=True, num_workers=2, pin_memory=True)


    print("Full DataSet image count:", len(ds))
    print("Train dataset: ", len(train_set), "Valid dataset: ", len(val_set),"Valid dataset: ",len(test_set))


    # block size
    block_sizes = [16,32,64]
    # number of layers
    num_layers =  [4,8,12]
    # kernel size
    kernels = [3,5,7]
    # average pool size
    AP = 20
    # number of epochs
    epochs = 100

    # learning rate
    lr = 0.001
    # weight decay
    wd = 0.0001
    # number of fc layers
    for B in block_sizes:
        for n in num_layers:
            for k in kernels:    
                F = int(n/3)   
                # create model and learner objects and train the model on the training set and evaluate it on the validation set
                model_dir = os.path.join(model_path,"B{}FC{}K{}AP{}".format(n,F,k,AP))
                os.mkdir(model_dir)
                model_name = os.path.join(model_dir,"B{}FC{}K{}AP{}".format(n,F,k,AP))
                model = ResNet(num_classes=4,num_channel=B,num_blocks=n,num_fc_layers=F,avg_pool_size=AP)
                # save best model path and name
                save_best_path = os.path.join(model_path,model_name)
                learn = Learner(train_dl, val_dl, model,labels_name=ds_labels,save_best=model_name,log_path=model_name)
                # load the best model
                learn.train_eval(epochs=epochs,lr=lr,wd = wd)

                learn.plot_metrics_macro(model_name + "metrics_mocro.svg")
                learn.plot_metrics_micro(model_name + "metrics_micro.svg")
                learn.plot_loss(model_name + "Loss.svg")
                
                # create confusion matrix for validation and test sets
                print("creating confusion matrix for validation")
                learn.predict(val_dl)
                learn.plotConfisuionMatrix(model_name + "confMatrixVal.svg")
                # create confusion matrix for validation and test sets
                print("creating confusion matrix for Test")
                learn.predict(test_dl)
                learn.plotConfisuionMatrix(model_name + "confMatrixTest.svg")
