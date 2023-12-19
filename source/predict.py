from classes import *
import pandas as pd

if __name__ == "__main__":
    path = "./DataSet/"
    model_path ="./models/"
    B = 16
    N =  8
    K = 5 
    FC = 2
    AP = 20
    epochs = 2
    k_folds = 2

    ds = ImageDataset(path)
    ds_files = ds.files

    ds_labels = list(ds.classes_dict.keys())
    
    test_dl = DataLoader(ds, 64, num_workers=2, pin_memory=True)
    
    model = ResNet(num_classes=4,num_channel=B,num_blocks=N,num_fc_layers=FC,avg_pool_size=AP)
    learn = Learner(None, None, model,labels_name=ds_labels,save_best=None,log_path=None)
    
    best_model = os.path.join("./best_model",os.listdir("./best_model")[0])
    learn.load(best_model)
    all_predictions, all_labels, all_probabilites = learn.predict(test_dl)

    df = pd.DataFrame(
        {"path": ds_files,
         "true_label":all_labels,
         "pred": all_predictions,
         "prob": all_probabilites}
    )
    df.to_csv("pred_resutls.csv")