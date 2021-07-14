import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import LeakyReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
plt.switch_backend("Agg")

class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path)
        df = df.drop(columns=["index"])
        # store the inputs and outputs
        self.X = df.values[:, :-1].astype("float32")
        self.y = df.values[:, -1].astype("float32")
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 50)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(50, 50)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(50, 1)
        xavier_uniform_(self.hidden3.weight)

        # print model state_dict
        print("model state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X

    # save model state_dict
    def save(self, path):
        torch.save(self.state_dict(), path)

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model, n_epochs=100):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(n_epochs):
        print("starting epoch", epoch)
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

def make_predictions(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    return predictions, actuals

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = make_predictions(test_dl, model)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse

def plot_predictions(test_dl, model, name):
    predictions, actuals = make_predictions(test_dl, model)
    fig = plt.figure(figsize=(5,5))
    plt.scatter(actuals, predictions, s=0.5, c="k")
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, c="gray", linestyle="--", alpha=0.75)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(name)
    plt.close()

# prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

def main():
    path = "dnn_data/EuropeanCall_GBM.100000smpl.csv"
    # prepare the data
    train_dl, test_dl = prepare_data(path)
    print(len(train_dl.dataset), len(test_dl.dataset))
    # define the network
    model = MLP(n_inputs=5)
    # train the model
    train_model(train_dl, model, n_epochs=2)

    # evaluate the model
    mse = evaluate_model(test_dl, model)
    print("MSE: %.6f, RMSE: %.6f" % (mse, np.sqrt(mse)))
    # # make a single prediction (expect class=1)
    row = test_dl.dataset[0][0]
    yhat = predict(row, model)
    print("Predicted: %.6f" % yhat)

    plot_predictions(test_dl, model, "test.png")

if __name__=="__main__":
    main()
