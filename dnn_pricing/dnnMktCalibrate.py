import torch
import scipy.optimize as spo
from sklearn.metrics import mean_squared_error
from dnnOptionPricer import *

S0 = 431.3399963378906
r = 0.02672004830382966
q = 0.02

dynamics = ["GBM", "HES", "MER"]
n_inputs = {"GBM": 5, "HES": 9, "MER": 8}
put_call = ["put", "call"]
model_dim = [120,120,120,120,120,1]
model_file = "model_500ep|120,120,120,120,120,1|l,l,l,l,l,i.pt"
option_data = "../bkts_data/option_chain_SPY_2021-07-16.csv"

def main():
    dyn = "HES"
    pc = "put"
    data = pd.read_csv(option_data)
    data = data[data["Put/Call"]==pc.capitalize()]
    K = data["Strike"].values
    T = data["Maturity (Year)"].values
    V = data["Market Price"].values
    n = len(data)
    model_path = "result/%s trained networks/%s/%s" % (dyn, pc, model_file)
    model = MLP(n_inputs[dyn], model_dim)
    model.load(model_path)
    def dnn_pricer(sig,kappa,theta,zeta,rho):
        inputs = torch.cat((torch.t(Tensor([S0/K,T])), Tensor([[r,q,sig,kappa,theta,zeta,rho]]*n)), dim=1)
        return K * model(inputs).detach().numpy().flatten()
    def objective(x):
        return np.sqrt(mean_squared_error(dnn_pricer(*x), V))
    init_vals = [0.2,1,0.04,0.5,-0.5]
    print("initial objective", objective(init_vals))
    # bounds = [(0,2), (0,100), (0,4), (0,2), (-1,1)]
    # opt_res = spo.minimize(objective, x0=init_vals, bounds=bounds, method="Newton-CG")
    for m in ["Nelder-Mead","Powell","COBYLA"]:
        print("using %s" % m)
        opt_res = spo.minimize(objective, x0=init_vals, method=m)
        params = opt_res.x
        print(params, objective(params))
        print(dnn_pricer(*params), V)

    # for dyn in dynamics:
    #     for pc in put_call:
    #         print("calibrating %s %s" % (dyn, pc))
    #         model_path = "result/%s trained networks/%s/%s" % (dyn, pc, model_file)
    #         model = MLP(n_inputs[dyn], model_dim)
    #         model.load(model_path)
    #         pass

if __name__=="__main__":
    main()
