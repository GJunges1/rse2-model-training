# This is the code for the 16-fold experiment
# I've written this one in pure python because I ran it on a cluster

import pandas as pd
import numpy as np

path = ""
df = pd.read_pickle(path + "df_All_Embeddings.pkl")
df_groups = pd.read_csv(path + "projectId.csv", header=None)

# criando datasets X e y
X = df["all_embeddings"].iloc[:].values
y = df["Estimate"].iloc[:].values
X = np.asarray([np.asarray(x).astype("float32") for x in X])
y = np.asarray(y).astype("float32")
groups = np.asarray(df_groups[0].iloc[:].values).astype("int32")
print(np.shape(X))
print(np.shape(y))
print(np.shape(groups))
del df, df_groups

"""caso seja necessário continuar de um ponto, descomente abaixo e MUDE: ``` N = ... ``` na função nested"""

acc_vector = {
    "mae": [],
    "mse": [],
    "mdae": [],
    "max_error": [],
    "r2_score": [],
}

# EXPERIMENTO DOIS (2)

# Note: This part of the notebook is due to I haved stopped the experiment run before its end,
# and continued from a specific iteration

partial_results = [
    {
        "mae": 2.262189,
        "mse": 11.59539,
        "mdae": 1.5764661,
        "max_error": 36.034897,
        "r2_score": -0.0476282143560276,
    },
    {
        "mae": 4.0535173,
        "mse": 37.1009,
        "mdae": 2.8459692,
        "max_error": 35.734406,
        "r2_score": -0.04754126016803051,
    },
    {
        "mae": 2.3056066,
        "mse": 9.832633,
        "mdae": 1.8499835,
        "max_error": 18.184679,
        "r2_score": -1.1427296469700767,
    },
    {
        "mae": 3.5396683,
        "mse": 38.697002,
        "mdae": 2.1473877,
        "max_error": 37.574234,
        "r2_score": 0.0966024033127233,
    },
    {
        "mae": 7.7454934,
        "mse": 274.97076,
        "mdae": 3.1135044,
        "max_error": 97.4672,
        "r2_score": 0.002476915647207978,
    },
    {
        "mae": 3.5271206,
        "mse": 23.018856,
        "mdae": 2.6851325,
        "max_error": 35.004513,
        "r2_score": -4.5907034705026275,
    },
]
for partial_i in partial_results:
    for key in partial_i.keys():
        acc_vector[key].append(partial_i[key])

print("resuming from:", acc_vector, "-----------------------------\n", sep="\n")

import keras
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    max_error,
)
from keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator

# from joblib import Parallel
from tqdm import tqdm
import gc
import numpy.ma as ma
import math


def desNaNify(x):
    if math.isnan(x):
        return (
            9999  # retorna um número grande ao invés de nan, para não dar erro no mse
        )
    return x


class MyEstimator(BaseEstimator):
    model = None

    def __init__(
        self,
        n1: int = 256,
        n2: int = 256,
        n3: int = 256,
        lr: float = 0.001,
        batch_size: int = 256,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
    ):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, X_train, y, X_valid, y_valid):
        self.input_layer = Input(shape=(2048,))
        # model = expand_dims(input_layer,axis=-1)
        self.model = Dense(self.n1, kernel_initializer="normal", activation="relu")(
            self.input_layer
        )  # (model)# Híperparâmetros [LSTM]: dropout, recurrent_dropout, número de nós
        self.model = Dense(self.n2, kernel_initializer="normal", activation="relu")(
            self.model
        )  # Híperparâmetros [DENSE]: Número de nós por camada
        self.model = Dense(self.n3, kernel_initializer="normal", activation="relu")(
            self.model
        )
        self.model = Dense(1, kernel_initializer="normal", activation="linear")(
            self.model
        )
        self.model = keras.Model(inputs=self.input_layer, outputs=self.model)

        adam = Adam(
            learning_rate=self.lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=None,
            decay=0.01,
            amsgrad=False,
        )
        # Híperparâmetros [Adam]: lr, beta1, beta2, epsilon, decay

        self.model.compile(
            loss="mse",
            optimizer=adam,
            metrics=["mae"],
            steps_per_execution=1,
        )

        earlystopping = EarlyStopping(
            monitor="val_mae",
            mode="auto",
            verbose=0,
            patience=10,
            restore_best_weights=True,
        )

        print(self.model.summary())
        self.result = self.model.fit(
            X_train,
            y,
            batch_size=self.batch_size,
            epochs=48,
            callbacks=[earlystopping],
            validation_data=(X_valid, y_valid),
            verbose=0,
        )
        del adam
        del earlystopping
        return self

    def predict(self, X):
        if self.model == None:
            print("ERRO!")
            return None
        y_hat = self.model.predict(X)
        return y_hat


def product_dict(**kwargs):
    import itertools

    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def my_iterable(config: dict, random_seed: int = None):
    import random

    if random_seed != None:
        random.seed(random_seed)
    aux = list(product_dict(**config))
    random.shuffle(aux)
    return aux


def my_accuracy(y_test, y_hat):
    return {
        "mae": mean_absolute_error(y_test, y_hat),
        "mse": mean_squared_error(y_test, y_hat),
        "mdae": median_absolute_error(y_test, y_hat),
        "max_error": max_error(y_test, y_hat),
        "r2_score": r2_score(y_test, y_hat),
    }


def nested(X, y, config={}, n_iter: int = None, k: int = 10):
    global acc_vector
    N = 6  # Última iteração executada
    thetamin_vector = []
    # acc_vector = {
    #     'mae': [],
    #     'mse': [],
    #     'mdae': [],
    #     'max_error': [],
    #     'r2_score': [],
    # }
    logo = LeaveOneGroupOut()  # divisão por projeto!
    outer_k = logo.get_n_splits(X, y, groups=groups)
    i = 0
    iterable_parameters = my_iterable(config=config, random_seed=123)[:n_iter]
    if n_iter != None and n_iter == len(iterable_parameters):
        print(
            "\n", (outer_k - N) * (n_iter * k + 1), " models will be fitted\n", sep=""
        )
    else:
        print(
            "\n",
            (outer_k - N) * (len(iterable_parameters) * k + 1),
            " models will be fitted\n",
            sep="",
        )
    print("------------------------------------------------\n")
    for tr_o, te_o in logo.split(X, groups=groups):
        i += 1
        if i > N:  # N is the last iteration the code successfully reported results
            print("\n[STARTED] OUTER_CV ", i, "/", outer_k, sep="")
            X_train_o, X_test_o = X[tr_o], X[te_o]
            y_train_o, y_test_o = y[tr_o], y[te_o]
            accmin = 0.0
            thetamin = None
            j = 0
            for theta in iterable_parameters:  # grid/random/hyperopt search
                j += 1
                print(
                    "\nCurrent parameter configuration being tested [",
                    j,
                    "/",
                    len(iterable_parameters),
                    "]:",
                    sep="",
                )
                print(theta, "\n", sep="")
                acc = 0.0
                cv_inner = KFold(n_splits=k, shuffle=True, random_state=2)
                l = 0
                for tr_i, te_i in tqdm(cv_inner.split(X_train_o)):
                    X_train_i, X_test_i = X_train_o[tr_i], X_train_o[te_i]
                    y_train_i, y_test_i = y_train_o[tr_i], y_train_o[te_i]

                    fit_params = {"X_valid": X_test_i, "y_valid": y_test_i}
                    model_i = MyEstimator(**theta).fit(
                        X_train_i, y_train_i, **fit_params
                    )  # fit
                    y_hat = model_i.predict(X_test_i)
                    y_hat = np.vectorize(lambda x: desNaNify(x))(y_hat)
                    print(
                        "\n",
                        y_hat[ma.masked_invalid(y_hat)._mask],
                        " ; ",
                        X_test_i[ma.masked_invalid(X_test_i)._mask],
                        sep="",
                        end="",
                    )
                    # evaluate the model
                    mse = mean_squared_error(y_test_i, y_hat)
                    acc += mse
                    l += 1
                    print("\nINNER_CV ", l, "/", k, " concluded", sep="")
                    del fit_params
                    del model_i
                    del y_hat
                    del mse
                    del X_train_i, X_test_i, y_train_i, y_test_i
                    del tr_i, te_i
                # END OF INNER LOOP
                print(
                    "Hiperparameter RESULTS [",
                    j,
                    "/",
                    len(iterable_parameters),
                    "]: ",
                    acc,
                    sep="",
                )
                if acc < accmin or j == 1:
                    accmin = acc
                    thetamin = theta
                del acc
                del theta
                del cv_inner
                gc.collect()
            # END OF RANDOMSEARCH LOOP
            fit_params = {"X_valid": X_test_o, "y_valid": y_test_o}
            model_o = MyEstimator(**thetamin).fit(X_train_o, y_train_o, **fit_params)
            y_hat = model_o.predict(X_test_o)
            aux = my_accuracy(y_test_o, y_hat)
            for key in acc_vector.keys():
                acc_vector[key].append(aux[key])
            thetamin_vector.append(thetamin)
            print("\n[FINISHED] OUTER_CV ", i, "/", outer_k, sep="")
            print("Iteration Results:")
            print(thetamin)
            print(aux)
            print("---------------------------------------")
            del aux
            del thetamin
            del y_hat
            del model_o
            del fit_params
            del X_train_o, X_test_o, y_test_o, y_train_o
    # END OF OUTER LOOP
    accfinal = dict()
    for key in acc_vector.keys():
        accfinal["mean_" + key] = np.mean(acc_vector[key])
        if key in ["r2_score", "mae", "mse"]:
            accfinal["std_" + key] = np.std(acc_vector[key])
    accfinal["train_results"] = acc_vector
    accfinal["thetas"] = thetamin_vector
    del acc_vector
    del thetamin_vector
    return accfinal


# END OF NESTED FUNCTION

config = {  # 2*5*2*9=180
    "n1": [512],  # 2
    "n2": [512],  # 1
    "n3": [512],  # 1
    "lr": [1e-5, 1e-4, 1e-3, 1e-2],  # 5
    "batch_size": [64, 128],  # 2
    "beta_1": [0.9, 0.95, 0.99],  # 3
    "beta_2": [0.9, 0.99, 0.999],  # 3
}
results = nested(X, y, config=config, n_iter=100, k=10)
print(
    "\n------------------------------------------\nNESTED CV RESULTS:",
    results,
    sep="\n",
)

import pickle
from datetime import datetime

now = datetime.now()
current_time = now.strftime("-%d%m%y-%H%M")

with open(path + "results" + current_time + ".pickle", "wb") as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("\nThe results were been written to a results.pckl file!")

# !pip3 install pickle5 --quiet
# import pickle5 as pickle
# with open('results-181022-1804.pickle', 'rb') as handle:
#   unserialized_data = pickle.load(handle)
# print(unserialized_data['thetas'][0])
