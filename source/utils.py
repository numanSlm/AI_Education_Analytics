import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid

def img_visualizer(df, row_index, rows=1, cols=1,save_name='',show_label=True):
    '''
    This Function creates a subplot of size(rows, cols) and visulaize the list of indexes of df
    the df should have a 'pixels' columns with 1 array shape of 2304 to rearrage it 48x48
    '''
    
    if type(row_index) != list:
        row_index = [row_index]
    assert len(row_index) == rows*cols,"number row index list must be equal to number of rows time number of columns"
    
    n = 0
    fig, axs = plt.subplots(rows, cols)
    for i in range(1,rows*cols+1):
            pixels = df["pixels"][row_index[n]]
            img = np.array(pixels.split(' '),dtype='int').reshape((48,48))
            plt.subplot(rows, cols, i)
            plt.imshow(img,cmap='gray')
            plt.axis('off')
            if show_label == True:
                plt.title(df["emotion"][row_index[n]])
            n+=1

    if save_name != '':
         fig.savefig(save_name)


def new_emotions(nautral,happy,surprise,sad,anger,disgust,fear):
    a1=np.greater(nautral, happy)
    a2=np.greater(nautral, surprise)
    a3=np.greater(nautral, sad)
    a4=np.greater(nautral, anger)
    a5=np.greater(nautral, disgust)
    a6=np.greater(nautral, fear)
    
    b1=np.greater(anger, happy)
    b2=np.greater(anger, surprise)
    b3=np.greater(anger, sad)
    b4=np.greater(anger, nautral)
    b5=np.greater(anger, disgust)
    b6=np.greater(anger, fear)

    if np.greater(anger,2) or (b1 and b2 and b3 and b4 and b5 and b6):    
        return "angry"
    if (a1 and a2 and a3 and a4 and a5 and a6) and(sad == 0) and (anger == 0):
        return "focused"
    if (a1 and a2 and a3 and a4 and a5 and a6) and nautral > 6:
        return "nautral"
    if (sad != 0) and (nautral != 0)and (fear == 0) and (happy == 0) and(anger+2 < sad):
        return "bored"

    else:
        return "other"

# show images in a grid format with nmax images per row 
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
# show a batch of images in a grid format with nmax images per row
def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break