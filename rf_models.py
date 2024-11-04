import os
import warnings
from collections import Counter

import dill
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier
from skopt import forest_minimize
from skopt.plots import plot_convergence, plot_evaluations
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args


DISCRETE_COLUMNS = [
    "GENDER",
    "RACE",
    "ETHNICITY",
    "REGION",
    "DIVISION",
    "age_group",
]
CONTINUOUS_COLUMNS = [
    "LOS",
    "DIAG_12M",
    "DIAG_12M_uniq",
    "DIAG",
    "DIAG_uniq",
    "vis_12M",
    "lab_12M",
    "lab_12M_n",
    "lab_ad",
    "lab_ad_n",
    "WBC",
    "RBC",
    "Creatinine",
    "HCT",
    "HGB",
    "BUN",
    "eGFR",
    "PLT",
    "AST",
    "SBP",
    "BMI",
    "RESP",
] 
TARGET_COLUMN = "Readmission"  
TARGETS = ["No", "Yes"]


def simple_imputer(values, imputer_path, strategy="mean", mode="train"):
    if mode == "train":
        imputer = SimpleImputer(strategy=strategy)  
        imputer.fit(values) 
        save_pkl(imputer_path, imputer)  
    else:
        imputer = load_pkl(imputer_path)  
    return imputer.transform(values)

def ordinal_encoder(values, encoder_path, mode="train"):
    if mode == "train":
        encoder = OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ) 
        encoder.fit(values) 
        save_pkl(encoder_path, encoder) 
    else:
        encoder = load_pkl(encoder_path)  
    return encoder.transform(values) + 1 


def standard_scaler(values, scaler_path, mode="train"):
    if mode == "train":
        scaler = StandardScaler() 
        scaler.fit(values)  
        save_pkl(scaler_path, scaler) 
    else:
        scaler = load_pkl(scaler_path)  
    return scaler.transform(values) 


def load_data(data_path, continuous_imputer_path, discrete_encoder_path, continuous_scaler_path):
    data = pd.read_csv(data_path)

    cols_labs = ["WBC", "RBC", "Creatinine", "HCT", "HGB", "BUN", "eGFR", "PLT", "AST"]
    cols_obs = ["SBP", "BMI", "RESP"]
    data = data.dropna(subset=cols_labs, how="all")
    data = data.dropna(subset=cols_obs, how="all")

    data[cols_labs + cols_obs] = simple_imputer(
        data[cols_labs + cols_obs].values,
        continuous_imputer_path,
        strategy="mean",
        mode="train",
    )
    data[DISCRETE_COLUMNS] = ordinal_encoder(
        data[DISCRETE_COLUMNS].values,
        discrete_encoder_path,
        mode="train",
    )
    data[CONTINUOUS_COLUMNS] = standard_scaler(
        data[CONTINUOUS_COLUMNS].values,
        continuous_scaler_path,
        mode="train",
    )

    X = data[DISCRETE_COLUMNS + CONTINUOUS_COLUMNS].values.tolist()  # 获取特征
    y = data[TARGET_COLUMN].apply(lambda x: TARGETS.index(x)).values.tolist()  # 获取标签

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def save_evaluate(y_test, y_test_pred, outputs_path):
   
    test_report = classification_report(
        y_test, y_test_pred, digits=4
    )  
    test_matrix = confusion_matrix(y_test, y_test_pred)  

    results = (
        "test classification report:\n" + test_report + "\nconfusion matrix\n" + str(test_matrix)
    )
    save_txt(outputs_path, results) 
    print(results) 


def plot_confusion_matrix(y_test, y_test_pred, output_path):
    matrix = confusion_matrix(y_test, y_test_pred) 
   
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100)
    sns.heatmap(matrix, annot=True, fmt=".0f", linewidths=0.5, square=True, cmap="Blues", ax=ax) 
    ax.set_title("Confusion matrix visualization")  
    ax.set_xlabel("Real targets")  
    ax.set_ylabel("Pred targets")  
    ax.set_xticks([x + 0.5 for x in range(len(TARGETS))], TARGETS, rotation=0) 
    ax.set_yticks([x + 0.5 for x in range(len(TARGETS))], TARGETS, rotation=0) 
    plt.tight_layout()  
    plt.savefig(output_path) 
    plt.close()  


def plot_roc(y_test, y_test_pred_score, output_path):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100)  
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        y_test_pred_score,
    ) 
    roc_auc = auc(false_positive_rate, true_positive_rate)  
    ax.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:0.4f}") 
    ax.plot([0, 1], [0, 1], "r--")  
    ax.set_xlabel("False Positive Rate") 
    ax.set_ylabel("True Positive Rate")  
    ax.set_title("Model ROC visualization") 
    plt.legend(loc="lower right") 
    plt.savefig(output_path)  
    plt.close() 


def importances_visualization(features_name, feature_importances, output_path):
    features_name, feature_importances = zip(
        *sorted(zip(features_name, feature_importances), key=lambda x: x[1], reverse=True)
    ) 

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100) 

    ax.bar(features_name, feature_importances, width=0.5)  
    for x, y in zip(range(len(features_name)), [round(item, 4) for item in feature_importances]):
        ax.text(x=x, y=y, s=y, ha="center", va="bottom")  

    ax.set_xticks(range(len(features_name)), features_name, rotation=90) 
    ax.set_xlabel("Features", fontsize=12)  
    ax.set_ylabel("Importance", fontsize=12)  
    ax.set_title("Analysis of characteristic important degree", fontsize=14) 

    plt.tight_layout() 
    plt.savefig(output_path)  
    plt.close()  


def bayes_optimization(
    X_train, X_test, y_train, y_test, bayes_model_path, bayes_acc_output_path, bayes_params_output_path, output_path
):
    spaces = [
        Integer(1, 1000, name="n_estimators"),
        Integer(1, 100, name="max_depth"),
        Integer(2, 100, name="min_samples_split"),
        Integer(1, 10, name="min_samples_leaf"),
        Real(0, 0.5, name="min_weight_fraction_leaf"),
        Real(0.01, 1, name="max_samples"),
    ]

    @use_named_args(spaces)
    def objective(**kwargs):
        print("参数详情:", kwargs)
        model = RandomForestClassifier(n_jobs=-1, random_state=42)  
        model.set_params(**kwargs)  
        model.fit(X_train, y_train)

        # use auc as reference function
        # test_score = -accuracy_score(y_test, model.predict(X_test)) 
        y_test_pred_score = model.predict_proba(X_test)[:, 1] 
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_score) 
        test_score = -auc(fpr, tpr)  # 计算AUC

        return test_score

    result = forest_minimize(objective, spaces, n_calls=100, random_state=42, verbose=True, n_jobs=-1)  
    save_pkl(bayes_model_path, result)  

    plot_convergence(result)  
    plt.tight_layout()  
    plt.savefig(bayes_acc_output_path)
    plt.close()  

    plot_evaluations(result)  
    plt.tight_layout()  
    plt.savefig(bayes_params_output_path) 
    # plt.show()  
    plt.close()  

    # save the optimization paramtertes
    pd.DataFrame(
        {
            "score": (-1 * result["func_vals"]).tolist(),
            "params": [{k: v for k, v in zip([space._name for space in spaces], x)} for x in result["x_iters"]],
        }
    ).to_excel(
        output_path, index=False
    )  

    best_params = {k: v for k, v in zip([space._name for space in spaces], result["x"])} 

    print("best paramaters:", best_params)  
    print("best score:", -1 * result["fun"])  

    return best_params  


def default_model_run():
    for sampler in ["original", "over_sampler"]:  
        
        root_dir = os.path.abspath(os.path.dirname(__file__)) 
        data_path = os.path.join(root_dir, "pt_ff.csv")  
        outputs_dir = os.path.join(root_dir, f"outputs_{sampler}_rf_default")  
        makedir(outputs_dir) 

        X_train, X_test, y_train, y_test = load_data(
            data_path,
            os.path.join(outputs_dir, "continuous_imputer.pkl"),
            os.path.join(outputs_dir, "discrete_encoder.pkl"),
            os.path.join(outputs_dir, "continuous_scaler.pkl"),
        ) 

        if sampler == "over_sampler":
            print("Current train samples:", Counter(y_train))
            sampler_model = RandomUnderSampler(sampling_strategy="auto", random_state=42) 
            X_train, y_train = sampler_model.fit_resample(X_train, y_train) 
            print("After train samples:", Counter(y_train))
        else:
            pass

        model = RandomForestClassifier(n_jobs=-1, random_state=42)  
        model.fit(X_train, y_train)
        save_pkl(os.path.join(outputs_dir, "model.pkl"), model)

        y_test_pred = model.predict(X_test) 
        save_evaluate(y_test, y_test_pred, os.path.join(outputs_dir, "evaluate.txt")) 
        plot_confusion_matrix(y_test, y_test_pred, os.path.join(outputs_dir, "confusion_matrix.png"))  

        y_test_pred_score = model.predict_proba(X_test)[:, 1] 
        plot_roc(y_test, y_test_pred_score, os.path.join(outputs_dir, "roc.png"))  

        importances_visualization(
            DISCRETE_COLUMNS + CONTINUOUS_COLUMNS,
            model.feature_importances_,
            os.path.join(outputs_dir, "importances.png"),
        ) 


def bayes_model_run():
    for sampler in ["original", "over_sampler"]:
        root_dir = os.path.abspath(os.path.dirname(__file__)) 
        data_path = os.path.join(root_dir, "pt_ff.csv") 
        outputs_dir = os.path.join(root_dir, f"outputs_{sampler}_rf_bayes") 
        makedir(outputs_dir)

        X_train, X_test, y_train, y_test = load_data(
            data_path,
            os.path.join(outputs_dir, "continuous_imputer.pkl"),
            os.path.join(outputs_dir, "discrete_encoder.pkl"),
            os.path.join(outputs_dir, "continuous_scaler.pkl"),
        )

        if sampler == "over_sampler":
            print("Current train samples:", Counter(y_train))
            sampler_model = RandomUnderSampler(sampling_strategy="auto", random_state=42) 
            X_train, y_train = sampler_model.fit_resample(X_train, y_train) 
            print("After train samples:", Counter(y_train))
        else:
            pass

        best_params = bayes_optimization(
            X_train,
            X_test,
            y_train,
            y_test,
            os.path.join(outputs_dir, "bayes.pkl"),
            os.path.join(outputs_dir, "bayes_acc.png"),
            os.path.join(outputs_dir, "bayes_params.png"),
            os.path.join(outputs_dir, "bayes_details.xlsx"),
        )
        
        model = RandomForestClassifier(n_jobs=-1, random_state=42) 
        model.set_params(**best_params) 
        model.fit(X_train, y_train)  
        save_pkl(os.path.join(outputs_dir, "model.pkl"), model) 

        y_test_pred = model.predict(X_test) 
        save_evaluate(y_test, y_test_pred, os.path.join(outputs_dir, "evaluate.txt"))  
        plot_confusion_matrix(y_test, y_test_pred, os.path.join(outputs_dir, "confusion_matrix.png"))  

        y_test_pred_score = model.predict_proba(X_test)[:, 1]  
        plot_roc(y_test, y_test_pred_score, os.path.join(outputs_dir, "roc.png"))  

        importances_visualization(
            DISCRETE_COLUMNS + CONTINUOUS_COLUMNS,
            model.feature_importances_,
            os.path.join(outputs_dir, "importances.png"),
        )  


if __name__ == "__main__":
    default_model_run()  
    bayes_model_run() 
