from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)


def accuracy(gts, preds):
    return accuracy_score(gts, preds) * 100


def macro_f1(gts, preds):
    return f1_score(gts, preds, average="macro") * 100


def report(gts, preds, target_names=None):
    return classification_report(gts, preds, target_names=target_names, digits=4)


def confusion(gts, preds):
    return confusion_matrix(gts, preds)
