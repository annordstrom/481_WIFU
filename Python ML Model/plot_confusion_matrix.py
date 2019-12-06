"""
Created on Thu Nov  7 16:04:40 2019

@author: matthew
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

#%% function to plot confusion matrix in a pretty manner
def pcm(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, name='confusion_matrix', directory=os.getcwd()):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel = 'True label',
           xlabel = 'Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    weight='bold' if i==j else 'normal')
    fig.tight_layout()
    file_name = directory + os.sep + name
    fig.savefig(file_name)