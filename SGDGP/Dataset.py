from libraries import *
​
​
class GetData:
    def __init__(self, len, type, seed):
        self.len = len
        self.type = type
        self.seed = seed
        np.random.seed(seed)
​
    def wing_weight(self):
        # http://www.sfu.ca/~ssurjano/wingweight.html
        n = self.len
        X = pd.DataFrame(
            dict(
                Sw=np.random.uniform(150, 200, size=n),
                Wfw=np.random.uniform(220, 300, size=n),
                A=np.random.uniform(6, 10, size=n),
                LamCaps=np.random.uniform(-10, 10, size=n) * np.pi / 180,
                q=np.random.uniform(16, 45, size=n),
                lam=np.random.uniform(0.5, 1, size=n),
                tc=np.random.uniform(0.08, 0.18, size=n),
                Nz=np.random.uniform(2.5, 6, size=n),
                Wdg=np.random.uniform(1700, 2500, size=n),
                Wp=np.random.uniform(0.025, 0.08, size=n),
            )
        )
​
        fact1 = 0.036 * X["Sw"] ** 0.758 * X["Wfw"] ** 0.0035
        fact2 = (X["A"] / ((np.cos(X["LamCaps"])) ** 2)) ** 0.6
        fact3 = X["q"] ** 0.006 * X["lam"] ** 0.04
        fact4 = 100 * X["tc"] / np.cos(X["LamCaps"]) ** (-0.3)
        fact5 = (X["Nz"] * X["Wdg"]) ** 0.49
​
        term1 = X["Sw"] * X["Wp"]
​
        y = fact1 * fact2 * fact3 * fact4 * fact5 + term1
        return X, y
​
    def bike(self):
        data = pd.read_csv("raw_data/bike.csv", index_col=[0], header=None)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y
​
    def protein(self):
        data = pd.read_csv("raw_data/protein.csv", header=None)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y
​
    def pm25(self):
        data = pd.read_csv('raw_data/pm25.csv', index_col=[0])
        data.columns = ['year', 'month', 'day', 'hour', 'pm25', 'dew_point', 'temperature',
               'pressure', 'wind_dir', 'wind_speed', 'hours_snow', 'hours_rain']
        l_enc = LabelEncoder()
        oh_enc = OneHotEncoder(sparse=False)
        data = data.dropna().reset_index(drop=True)
        data.wind_dir = l_enc.fit_transform(data.wind_dir)
        temp = pd.DataFrame(oh_enc.fit_transform(data.wind_dir.values.reshape(-1, 1)), columns=l_enc.classes_)
        data = data.drop('wind_dir', axis=1)
        data = data.join(temp)
        X = data.drop('pm25', axis=1)
        y = data['pm25']
        return X, y
        
    def energy(self):
        data = pd.read_csv("raw_data/energy.csv", index_col=[0]).reset_index(drop=True)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        return X, y
​
    def otl_circuit(self):
        # http://www.sfu.ca/~ssurjano/otlcircuit.html
        n = self.len
        X = pd.DataFrame(
            dict(
                Rb1=np.random.uniform(50, 150, size=n),
                Rb2=np.random.uniform(25, 70, size=n),
                Rf=np.random.uniform(0.5, 3, size=n),
                Rc1=np.random.uniform(1.2, 2.5, size=n) * np.pi / 180,
                Rc2=np.random.uniform(0.25, 1.2, size=n),
                beta=np.random.uniform(50, 300, size=n),
            )
        )
​
        Vb1 = 12 * X["Rb2"] / (X["Rb1"] + X["Rb2"])
        term1a = (Vb1 + 0.74) * X["beta"] * (X["Rc2"] + 9)
        term1b = X["beta"] * (X["Rc2"]) + X["Rf"]
        term1 = term1a / term1b
​
        term2a = 11.35 * X["Rf"]
        term2b = X["beta"] * (X["Rc2"] + 9) + X["Rf"]
        term2 = term2a / term2b
​
        term3a = 0.74 * X["Rf"] * X["beta"] * (X["Rc2"] + 9)
        term3b = (X["beta"] * (X["Rc2"] + 9) + X["Rf"]) * X["Rc1"]
        term3 = term3a / term3b
​
        Vm = term1 + term2 + term3
        return X, Vm
​
    def borehole(self):
        # http://www.sfu.ca/~ssurjano/borehole.html
        n = self.len
        X = pd.DataFrame(
            dict(
                rw=np.random.uniform(0.05, 0.15, size=n),
                r=np.random.uniform(100, 50000, size=n),
                Tu=np.random.uniform(63070, 115600, size=n),
                Hu=np.random.uniform(990, 1110, size=n) * np.pi / 180,
                Tl=np.random.uniform(63.1, 116, size=n),
                Hl=np.random.uniform(700, 820, size=n),
                L=np.random.uniform(1120, 1680, size=n),
                Kw=np.random.uniform(9855, 12045, size=n),
            )
        )
        frac1 = 2 * np.pi * X["Tu"] * (X["Hu"] - X["Hl"])
​
        frac2a = (
            2 * X["L"] * X["Tu"] / (np.log(X["r"] / X["rw"]) * X["rw"] ** 2 * X["Kw"])
        )
        frac2b = X["Tu"] / X["Tl"]
        frac2 = np.log(X["r"] / X["rw"]) * (1 + frac2a + frac2b)
​
        y = frac1 / frac2
        return X, y
​
    def griewank(self):
        # https://www.sfu.ca/~ssurjano/griewank.html
        n = self.len
        X = pd.DataFrame(
            dict(
                x1=np.random.uniform(-600, 600, size=n),
                x2=np.random.uniform(-600, 600, size=n),
                x3=np.random.uniform(-600, 600, size=n),
                x4=np.random.uniform(-600, 600, size=n),
                x5=np.random.uniform(-600, 600, size=n),
                x6=np.random.uniform(-600, 600, size=n),
            )
        )
        sum = 0
        prod = 1
        i = 1
        for ii in X.columns:
            xi = X[ii]
            sum += (xi**2) / 4000
            prod *= np.cos(xi / np.sqrt(i))
            i += 1
​
        y = sum - prod + 1
        return X, y
​
    def levy(self):
        # https://www.sfu.ca/~ssurjano/levy.html
        n = self.len
        X = pd.DataFrame(
            dict(
                x1=np.random.uniform(-10, 10, size=n),
                x2=np.random.uniform(-10, 10, size=n),
                x3=np.random.uniform(-10, 10, size=n),
                x4=np.random.uniform(-10, 10, size=n),
            )
        )
        w = []
        for ii in X.columns:
            w.append(1 + (X[ii] - 1) / 4)
​
        term1 = (np.sin(np.pi * w[0])) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)
​
        sum = 0
        for ii in range(len(X.columns) - 1):
            wi = w[ii]
            new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
            sum = sum + new
​
        y = term1 + sum + term3
        return X, y
​
    def make_data(self):
        X, y = eval("self." + self.type + "()")
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=self.seed
        )
        X_train = pd.DataFrame(scaler.fit_transform(X_train))
        X_test = pd.DataFrame(scaler.transform(X_test))
        y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
        y_test = scaler.transform(np.array(y_test).reshape(-1, 1))
        # X_train.to_csv("./X_train.csv.gz")
        # X_test.to_csv("./X_test.csv.gz")
        # pd.DataFrame(y_train).to_csv("./y_train.csv.gz")
        # pd.DataFrame(y_test).to_csv("./y_test.csv.gz")
        # X_train, X_test, y_train, y_test = torch.tensor(np.array([X_train])).squeeze(),torch.tensor(np.array([X_test])).squeeze(),torch.tensor(np.array([y_train])).squeeze(),torch.tensor(np.array([y_test])).squeeze()
        return X_train, X_test, y_train, y_test
​
    # For Bike Data: https://archive.ics.uci.edu/ml/machine-learning-databases/00275/ (hour.csv from this directory)
​
​
class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"
​
    def __init__(self, X, y):
        "Initialization"
        self.labels = y
        self.list_IDs = range(len(X))
        self.X = X
​
    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)
​
    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        ID = self.list_IDs[index]
​
        # Load data and get label
        X = self.X[ID]
        y = self.labels[ID]
​
        return X, y
