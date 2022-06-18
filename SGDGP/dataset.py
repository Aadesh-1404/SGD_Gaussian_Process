import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyDOE import *
from smt.sampling_methods import LHS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class dataset:
    def __init__(self, seed):
        self.seed = seed

    def levy(self):
        xlimits = np.array([[-10.0, 10.0], [-10.0, 10.0],
                           [-10.0, 10.0], [-10.0, 10.0]])
        random_state = np.random.seed(self.seed)
        sampling = LHS(xlimits=xlimits, random_state=self.seed)
        num = 10000
        # print(num)

        x = sampling(num)
        X = pd.DataFrame(
            dict(
                x1=x[0],
                x2=x[1],
                x3=x[2],
                x4=x[3],
            )
        )
        # print(0)
        w = []
        for ii in X.columns:
            w.append(1 + (X[ii] - 1) / 4)

        term1 = (np.sin(np.pi * w[0])) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)

        sum = 0
        for ii in range(len(X.columns) - 1):
            wi = w[ii]
            new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
            sum = sum + new
        y = term1 + sum + term3
        scaler = StandardScaler()

        X = pd.DataFrame(scaler.fit_transform(X))
        y = scaler.fit_transform(np.array(y).reshape(-1, 1))
        X.to_csv("Datasets/Levy/X.csv.gz")
        pd.DataFrame(y).to_csv("Datasets/Levy/y.csv.gz")

    def griewank(self):
        xlimits = np.array([[-600.0, 600.0], [-600.0, 600.0], [-600.0, 600.0],
                           [-600.0, 600.0], [-600.0, 600.0], [-600.0, 600.0]])

        random_state = np.random.seed(self.seed)
        sampling = LHS(xlimits=xlimits, random_state=self.seed)
        num = 10000
        # print(num)

        x = sampling(num)
        X = pd.DataFrame(dict(
            x1=x[0],
            x2=x[1],
            x3=x[2],
            x4=x[3],
            x5=x[4],
            x6=x[5], ))
        sum = 0
        prod = 1
        i = 1
        for ii in X.columns:
            xi = X[ii]
            sum += (xi**2) / 4000
            prod *= np.cos(xi / np.sqrt(i))
            i += 1

        y = sum - prod + 1
        scaler = StandardScaler()

        X = pd.DataFrame(scaler.fit_transform(X))
        y = scaler.fit_transform(np.array(y).reshape(-1, 1))
        X.to_csv("Datasets/griewank/X.csv.gz")
        pd.DataFrame(y).to_csv("Datasets/griewank/y.csv.gz")

    def wing_weight(self):
        num = 2000000
        xlimits = np.array([[150, 200], [220, 300], [6, 10], [-10, 10], [16, 45],
                            [0.5, 1], [0.08, 0.18], [2.5, 6], [1700, 2500], [0.025, 0.08]])
        sampling = LHS(xlimits=xlimits, random_state=self.seed)
        # print(num)

        x = sampling(num)
        # X = pd.DataFrame(x)
        X = pd.DataFrame(
            dict(
                Sw=x[0],
                Wfw=x[1],
                A=x[2],
                LamCaps=x[3] * np.pi / 180,
                q=x[4],
                lam=x[5],
                tc=x[6],
                Nz=x[7],
                Wdg=x[8],
                Wp=x[9],
            )
        )

        fact1 = 0.036 * X["Sw"] ** 0.758 * X["Wfw"] ** 0.0035
        fact2 = (X["A"] / ((np.cos(X["LamCaps"])) ** 2)) ** 0.6
        fact3 = X["q"] ** 0.006 * X["lam"] ** 0.04
        fact4 = 100 * X["tc"] / np.cos(X["LamCaps"]) ** (-0.3)
        fact5 = (X["Nz"] * X["Wdg"]) ** 0.49

        term1 = X["Sw"] * X["Wp"]

        y = fact1 * fact2 * fact3 * fact4 * fact5 + term1
        scaler = StandardScaler()

        X = pd.DataFrame(scaler.fit_transform(X))
        y = scaler.fit_transform(np.array(y).reshape(-1, 1))
        X.to_csv("Datasets/wing_weight/X.csv.gz")
        pd.DataFrame(y).to_csv("Datasets/wing_weight/y.csv.gz")

    def otl_circuit(self):
        num = 2000000
        xlimits = np.array([[50, 150], [25, 70], [0.5, 3], [
            1.2, 2.5], [0.25, 1.2], [50, 300]])
        sampling = LHS(xlimits=xlimits, random_state=self.seed)
        # print(num)

        x = sampling(num)
        X = pd.DataFrame(
            dict(
                Rb1=x[0],
                Rb2=x[1],
                Rf=x[2],
                Rc1=x[3] * np.pi / 180,
                Rc2=x[4],
                beta=x[5],
            )
        )

        Vb1 = 12 * X["Rb2"] / (X["Rb1"] + X["Rb2"])
        term1a = (Vb1 + 0.74) * X["beta"] * (X["Rc2"] + 9)
        term1b = X["beta"] * (X["Rc2"]) + X["Rf"]
        term1 = term1a / term1b

        term2a = 11.35 * X["Rf"]
        term2b = X["beta"] * (X["Rc2"] + 9) + X["Rf"]
        term2 = term2a / term2b

        term3a = 0.74 * X["Rf"] * X["beta"] * (X["Rc2"] + 9)
        term3b = (X["beta"] * (X["Rc2"] + 9) + X["Rf"]) * X["Rc1"]
        term3 = term3a / term3b

        Vm = term1 + term2 + term3

        scaler = StandardScaler()

        X = pd.DataFrame(scaler.fit_transform(X))
        y = scaler.fit_transform(np.array(Vm).reshape(-1, 1))
        X.to_csv("Datasets/otl_circuit/X.csv.gz")
        pd.DataFrame(Vm).to_csv("Datasets/otl_circuit/y.csv.gz")

    def borehole(self):
        num = 1000000

        xlimits = np.array([[0.05, 0.15], [100, 50000], [63070, 115600], [990, 1110], [
            63.1, 116], [700, 820], [1120, 1680], [9855, 12045]])
        sampling = LHS(xlimits=xlimits, random_state=self.seed)
        # print(num)

        x = sampling(num)
        X = pd.DataFrame(
            dict(
                rw=x[0],
                r=x[1],
                Tu=x[2],
                Hu=x[3] * np.pi / 180,
                Tl=x[4],
                Hl=x[5],
                L=x[6],
                Kw=x[7],
            )
        )
        frac1 = 2 * np.pi * X["Tu"] * (X["Hu"] - X["Hl"])

        frac2a = (
            2 * X["L"] * X["Tu"] / (np.log(X["r"] / X["rw"])
                                    * X["rw"] ** 2 * X["Kw"])
        )
        frac2b = X["Tu"] / X["Tl"]
        frac2 = np.log(X["r"] / X["rw"]) * (1 + frac2a + frac2b)

        y = frac1 / frac2
        scaler = StandardScaler()

        X = pd.DataFrame(scaler.fit_transform(X))
        y = scaler.fit_transform(np.array(y).reshape(-1, 1))
        X.to_csv("Datasets/borehole/X.csv.gz")
        pd.DataFrame(y).to_csv("Datasets/borehole/y.csv.gz")


seed = 0
make_data = dataset(seed)
make_data.levy()
make_data.griewank()
make_data.borehole()
make_data.wing_weight()
make_data.otl_circuit()
