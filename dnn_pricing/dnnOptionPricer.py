import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Identity
from torch.nn import LeakyReLU
from torch.nn import Module
from torch.nn import ModuleList
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
plt.switch_backend("Agg")

load_folder = "dnn_data/"
data_folder = "test-0717/"

def smoother(y, box_pts):
    y_smooth = y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth

class CSVDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        df = df.drop(columns=["index"])
        self.X = df.values[:, :-1].astype("float32")
        self.y = df.values[:, -1].astype("float32")
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.3):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

class MLP(Module):
    def __init__(self, n_inputs, model_dim, act_list=[]):
        super(MLP, self).__init__()
        self.depth = len(model_dim)
        if not act_list: act_list = [LeakyReLU()]
        if len(act_list) == 1: act_list = act_list * (self.depth-1) + [Identity()]
        self.name = self.get_name(model_dim, act_list)
        self.act = act_list
        self.hidden = ModuleList()
        print("model activations:", self.act)
        for i in range(self.depth):
            nodes_in = n_inputs if i == 0 else model_dim[i-1]
            nodes_out = model_dim[i]
            self.hidden.append(Linear(nodes_in, nodes_out))
        for layer in self.hidden:
            xavier_uniform_(layer.weight)
        print("model state_dict:", "None" if not self.state_dict() else "")
        for param_tensor in self.state_dict():
            print(param_tensor, ">>", self.state_dict()[param_tensor].size())

    def get_name(self, model_dim, act_list):
        act_abrv = {
            "Identity":     "i",
            "Sigmoid":      "s",
            "Tanh":         "t",
            "ReLU":         "r",
            "LeakyReLU":    "l",
        }
        name = ",".join(map(str, model_dim)) + "|" + \
            ",".join(map(lambda x: act_abrv[str(x).split("(")[0]], act_list))
        return name

    def forward(self, X):
        for i in range(self.depth):
            X = self.hidden[i](X)
            X = self.act[i](X)
        return X

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=100, shuffle=True)
    test_dl = DataLoader(test, batch_size=1000, shuffle=False)
    print("train/test dataset size:", len(train_dl.dataset), len(test_dl.dataset))
    return train_dl, test_dl

def train_model(train_dl, model, n_epochs=100, vald_set=None,
    lr=0.01, momentum=0.9,
    save_log=False, log_name="train_loss.log",
    plot_loss=False, plot_name="train_loss.png"):
    loss_log = dict()
    if vald_set:
        loss_vald = dict()
        inputs_vald, targets_vald = vald_set
        actual_vald = targets_vald.numpy()
        actual_vald = actual_vald.reshape((len(actual_vald), 1))
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(n_epochs):
        epoch_name = "epoch-%d" % epoch
        print("running", epoch_name, end="")
        loss_log[epoch_name] = list()
        loss_batch0 = 1
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            loss_log[epoch_name].append(loss.item())
            if i == 0: loss_batch0 = loss.item()
            if i % 50 == 0: print(" >> %.2f%%" % (loss.item() / loss_batch0 * 100), end="")
        if vald_set:
            yhat_vald = model(inputs_vald)
            yhat_vald = yhat_vald.detach().numpy()
            loss_vald[epoch_name] = mean_squared_error(actual_vald, yhat_vald)
        print(" >>", epoch_name, "completes")
    if save_log: pd.DataFrame.from_dict(loss_log).to_csv(log_name, index=False)
    if plot_loss:
        box_pts = 100
        loss_val = np.log(np.concatenate(list(loss_log.values())))
        loss_idx = np.arange(len(loss_val))
        n_batch = len(loss_val) // n_epochs # mini-batches per epoch
        fig = plt.figure(figsize=(5,5))
        plt.scatter(loss_idx, loss_val, s=0.5, c="k")
        plt.plot(loss_idx[box_pts:-box_pts], smoother(loss_val, 100)[box_pts:-box_pts], c="r", linestyle="--", alpha=0.75, label="train loss")
        if vald_set: plt.plot(np.arange(1, n_epochs+1)*n_batch, np.log(list(loss_vald.values())), c="b", linestyle="--", alpha=0.75, label="vald loss")
        plt.title("model: %s\nn_epochs: %d, n_batch: %d" % (model.name, n_epochs, n_batch))
        plt.xlim([0, 1.01*loss_idx[-1]])
        plt.xlabel("Mini-batch")
        plt.ylabel("Log-loss function")
        plt.legend()
        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close()
    return loss_log

def make_predictions(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    return predictions, actuals

def evaluate_model(test_dl, model):
    predictions, actuals = make_predictions(test_dl, model)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = 1-mse/np.var(actuals)
    eval = {"MSE": mse, "RMSE": rmse, "R2": r2}
    print("model eval:", eval)
    return eval

def plot_predictions(test_dl, model, plot_name="pred.png"):
    predictions, actuals = make_predictions(test_dl, model)
    fig = plt.figure(figsize=(5,5))
    plt.scatter(actuals, predictions, s=0.5, c="k")
    lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, c="gray", linestyle="--", alpha=0.75)
    plt.title("model: %s" % model.name)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(plot_name)
    plt.close()

def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat

def main1():
    # prototype: <120,120,120,120,120,1|l,l,l,l,l,i>
    n_inputs = 5
    n_epochs = 500
    depth = 6
    width = 120
    model_args = {
        "n_inputs": n_inputs,
        "n_epochs": n_epochs,
        "width":    width,
        "depth":    depth,
    }
    data_path = load_folder+"EuropeanCall_GBM.100000smpl.csv"
    train_dl, test_dl = prepare_data(data_path)
    model_args["model_dim"] = [model_args["width"]] * (model_args["depth"]-1) + [1]
    model = MLP(n_inputs=model_args["n_inputs"], model_dim=model_args["model_dim"])
    model_name = str(model_args["n_epochs"]) + "ep|" + model.name
    print("running model <%s>" % model_name)
    train_loss = train_model(train_dl, model, lr=0.1,
        n_epochs=model_args["n_epochs"], vald_set=next(iter(test_dl)),
        save_log=True, log_name=data_folder+"loss_%s.log"%model_name,
        plot_loss=True, plot_name=data_folder+"loss_%s.png"%model_name)
    eval = evaluate_model(test_dl, model)
    plot_predictions(test_dl, model,
        plot_name=data_folder+"pred_%s.png"%model_name)
    model.save(data_folder+"model_%s.pt"%model_name)

def main2():
    n_inputs = 5
    n_epochs = 200
    depth0, depth1 = 3, 7
    width0, width1, delta_width = 20, 140, 20
    data_path = load_folder+"EuropeanCall_GBM.100000smpl.csv"
    train_dl, test_dl = prepare_data(data_path)
    eval_log = dict()
    for depth in range(depth0, depth1):
        for width in range(width0, width1, delta_width):
            model_args = {
                "n_inputs": n_inputs,
                "n_epochs": n_epochs,
                "width":    width,
                "depth":    depth,
            }
            model_args["model_dim"] = [model_args["width"]] * (model_args["depth"]-1) + [1]
            model = MLP(n_inputs=model_args["n_inputs"], model_dim=model_args["model_dim"])
            model_name = str(model_args["n_epochs"]) + "ep|" + model.name
            print("running model <%s>" % model_name)
            train_loss = train_model(train_dl, model, n_epochs=model_args["n_epochs"],
                save_log=True, log_name=data_folder+"loss_%s.log"%model_name,
                plot_loss=True, plot_name=data_folder+"loss_%s.png"%model_name)
            eval = evaluate_model(test_dl, model)
            plot_predictions(test_dl, model,
                plot_name=data_folder+"pred_%s.png"%model_name)
            eval_log[model_name] = eval
    pd.DataFrame.from_dict(eval_log).T.to_csv(data_folder+"eval.csv")

if __name__=="__main__":
    main1()
    # main2()
