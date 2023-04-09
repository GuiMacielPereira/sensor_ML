import numpy as np
import torch
import sklearn
from peratouch.data import Data
from peratouch.preprocessing import plot

class Results:
    def __init__(self, Data : Data, model):
        self.Data = Data
        self.model = model

    def acc_tr(self):
        return acc(self.model, self.Data.xtr, self.Data.ytr)

    def acc_val(self):
        return acc(self.model, self.Data.xv, self.Data.yv)

    def acc_te(self):
        return acc(self.model, self.Data.xte, self.Data.yte)

    def matthews_corrcoef_te(self):
        return matthews_corrcoef(self.model, self.Data.xte, self.Data.yte)

    def loss_val(self, criterion):
        with torch.no_grad():    # Each time model is called, need to avoid updating the weights
            return criterion(self.model(self.Data.xv), self.Data.yv).item()
 
    def loss_tr(self, criterion):
        with torch.no_grad():    # Each time model is called, need to avoid updating the weights
            return criterion(self.model(self.Data.xtr), self.Data.ytr).item()

    def test_metrics(self, report=False, conf_matrix=False):
        print("\nTest dataset metrics:")
        print(f"Overall Accuracy = {self.acc_te()*100:.1f}%, Matthews Corr Coef = {self.matthews_corrcoef_te():.2f}")
        print("\n")

        preds = self.model(self.Data.xte).data.max(1)[-1] 
        if report:
            print(sklearn.metrics.classification_report(self.Data.yte, preds))
        if conf_matrix:
            sklearn.metrics.ConfusionMatrixDisplay.from_predictions(self.Data.yte, preds)

    # TODO: Function below is working but needs adding labels
    def find_most_uncertain_preds(self):
        with torch.no_grad():
            cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
            out = self.model(self.Data.xte)
            loss_vals = cross_entropy(out, self.Data.yte)
            idxs_min = loss_vals.argsort(descending=True)     # First elements are the ones with higher losses 
            worst_preds = self.Data.xte[idxs_min].detach().cpu().numpy()
            plot(worst_preds)

def acc(model, x, y):
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out.data, 1)
        return (pred==y).float().mean()        # Pass bool to float and compute mean, doesn't leave device

def matthews_corrcoef(model, x, y):
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out.data, 1)      # Tensor in device
        return sklearn.metrics.matthews_corrcoef(y, pred)    # Works with tensors in device

