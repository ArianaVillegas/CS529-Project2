import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def cross_validation_split(X, y, model, n_splits=10):
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    scores = []
    for train_ix, val_ix in cv.split(X):
        X_train, X_val = X[train_ix, :], X[val_ix, :]
        y_train, y_val = y[train_ix], y[val_ix]
        y_train = y_train.toarray()
        y_val = y_val.toarray()
        model.train(X_train, y_train)
        y_val_pred = model.eval(X_val)
        scores.append((y_val==y_val_pred).mean())
    return scores


def plot_confussion_matrix(y, pred, labels, filename):
    cm = confusion_matrix(y, pred, normalize='true')
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.matshow(cm, cmap=plt.cm.Spectral_r)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=round(cm[i, j], 2), va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xticks(range(len(labels)), [''] + labels, rotation='vertical', fontsize=18)
    plt.yticks(range(len(labels)), [''] + labels, fontsize=18)
    plt.tight_layout()
    plt.savefig(filename)