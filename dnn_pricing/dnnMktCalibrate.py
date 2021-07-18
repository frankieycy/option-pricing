import torch
import scipy.optimize as spo
from sklearn.metrics import mean_squared_error
from dnnOptionPricer import *

data_folder = "test-0718/"

args = {
    "S0": 431.3399963378906,
    "r": 0.02672004830382966,
    "q": 0.02,
    "M_range": [0.8, 1.2],
    "dynamics": ["GBM", "HES", "MER"],
    "n_inputs": {"GBM": 5, "HES": 9, "MER": 8},
    "put_call": ["put", "call"],
    "model_dim": [120,120,120,120,120,1],
    "model_file": "model_500ep|120,120,120,120,120,1|l,l,l,l,l,i.pt",
    "option_data": "../bkts_data/option_chain_SPY_2021-07-16.csv",
    "init_vals": {"GBM": [0.2], "HES": [0.2,1,0.04,0.5,-0.5], "MER": [0.2,1,0.1,0.2]}
}

output_cols = [
    "Contract Name",
    "Type",
    "Put/Call",
    "Strike",
    "Maturity (Year)",
    "Market Price",
    "DNN Price (Nelder-Mead)",
    "DNN Price (Powell)",
    "DNN Price (COBYLA)",
    "Implied Vol",
    "Delta",
    "Gamma",
    "Vega",
    "Rho",
    "Theta"
]

def calibrate_dnn(dyn, pc):
    print("calibrating %s %s" % (dyn, pc))
    data = pd.read_csv(args["option_data"])
    data = data[data["Put/Call"]==pc.capitalize()]
    K = data["Strike"].values
    M = args["S0"]/K
    data = data[(M >= args["M_range"][0]) & (M <= args["M_range"][1])]
    K = data["Strike"].values
    T = data["Maturity (Year)"].values
    V = data["Market Price"].values
    n = len(data)
    print("mkt data to calibrate:")
    print(data.head(10))
    MTtensor = torch.t(Tensor([args["S0"]/K,T]))
    model_path = "result/%s trained networks/%s/%s" % (dyn, pc, args["model_file"])
    model = MLP(args["n_inputs"][dyn], args["model_dim"])
    model.load(model_path)
    def dnn_pricer(x):
        inputs = torch.cat((MTtensor, Tensor([np.concatenate(([args["r"],args["q"]],x))]*n)), dim=1)
        return K * model(inputs).detach().numpy().flatten()
    def objective(x):
        return np.sqrt(mean_squared_error(dnn_pricer(x), V))
    init_vals = args["init_vals"][dyn]
    print("initial objective:", objective(init_vals))
    opt_log = dict()
    for m in ["Nelder-Mead", "Powell", "COBYLA"]:
        print("using method %s" % m)
        opt_res = spo.minimize(objective, x0=init_vals, method=m)
        params = opt_res.x
        data["DNN Price (%s)" % m] = dnn_pricer(params)
        opt_log[m] = {
            "opt_res": opt_res,
            "params": params
        }
        print("params:", params)
        print("objective:", objective(params))
    data = data[output_cols]
    data.to_csv(data_folder + "dnn_%s_%s.csv" % (dyn, pc), index=False)
    print()

def main():
    for dyn in args["dynamics"]:
        for pc in args["put_call"]:
            calibrate_dnn(dyn, pc)

if __name__=="__main__":
    main()
