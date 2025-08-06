import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="NGBoost SHAP Interpretation")
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                        help='Directory containing the dataset CSV files')
    parser.add_argument('--dataset_type', type=str, choices=['full', 'half'], default='full',
                        help='Dataset type: "full" or "half"')
    parser.add_argument('--suffix', type=str, default='mat_imputation',
                        help='Training dataset suffix')
    return parser.parse_args()


def load_model_and_scaler(suffix, dataset_type):
    model_folder = os.path.join("models", f"models_{dataset_type}")
    model_path = os.path.join(model_folder, f"best_ngb_default_model_{suffix}.pkl")
    scaler_path = os.path.join(model_folder, f"scaler_{suffix}.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler for {suffix} not found in {model_folder}.")
        return None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def load_training_data(suffix, dataset_dir, dataset_type, features):
    train_file = os.path.join(dataset_dir, f"Li-rich_train_{suffix}_{dataset_type}.csv")
    df = pd.read_csv(train_file)
    X = df[features]
    y = df['mean_primary_particle_size']
    return X, y


def plot_feature_importance(model, features, suffix, dataset_type):
    feature_importance_loc = model.feature_importances_[0]
    feature_importance_scale = model.feature_importances_[1]
    
    df_loc = pd.DataFrame({'feature': features, 'importance': feature_importance_loc})
    df_loc.sort_values('importance', ascending=False, inplace=True)
    df_scale = pd.DataFrame({'feature': features, 'importance': feature_importance_scale})
    df_scale.sort_values('importance', ascending=False, inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Feature Importance for Distribution Parameters", fontsize=17)
    sns.barplot(x='importance', y='feature', data=df_loc, color="skyblue", ax=ax1)
    ax1.set_title('loc (Mean)')
    sns.barplot(x='importance', y='feature', data=df_scale, color="skyblue", ax=ax2)
    ax2.set_title('scale (Uncertainty)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("plots", exist_ok=True)
    out_file = os.path.join("plots", f"Feature_importance_{suffix}_{dataset_type}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {out_file}")


def plot_shap_summary(model, X, features, suffix, dataset_type, param='loc'):
    model_output = 0 if param == 'loc' else 1
    # The TreeExplainer uses your ensemble of decision trees.
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output=model_output)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    os.makedirs("plots", exist_ok=True)
    out_file = os.path.join("plots", f"SHAP_summary_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot for {param} saved to {out_file}")


def plot_shap_advanced(model, X, features, suffix, dataset_type, param='loc'):
    model_output = 0 if param == 'loc' else 1
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output=model_output)
    shap_values = explainer.shap_values(X)
  
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    os.makedirs("plots", exist_ok=True)
    summary_file = os.path.join("plots", f"SHAP_summary_adv_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Advanced SHAP summary plot for {param} saved to {summary_file}")
    

    mean_abs_shap = np.abs(shap_values).mean(0)
    top_feature_idx = np.argmax(mean_abs_shap)
    top_feature = features[top_feature_idx]

    plt.figure()
    shap.dependence_plot(top_feature, shap_values, X, feature_names=features,
                           interaction_index="auto", show=False)
    os.makedirs("plots", exist_ok=True)
    depend_file = os.path.join("plots", f"SHAP_dependence_{param}_{top_feature}_{suffix}_{dataset_type}.png")
    plt.savefig(depend_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Advanced SHAP dependence plot for {param} and feature '{top_feature}' saved to {depend_file}")


def plot_shap_interaction(model, X, features, suffix, dataset_type, param='loc'):
    model_output = 0 if param == 'loc' else 1
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output=model_output)
    interaction_values = explainer.shap_interaction_values(X)
    
    plt.figure()
    features_np = np.array(features)
    shap.summary_plot(
        interaction_values,
        X,
        feature_names=features_np,
        show=False,
        plot_size=(14, 6),
        max_display=len(features_np)
    )
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out_file = os.path.join("plots", f"SHAP_interaction_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP interaction plot for {param} saved to {out_file}")

def main():
    args = parse_args()
    
    features = [
        'Li_fraction_in_TM_layer', 'Ni_fraction', 'Co_fraction', 'Mn_fraction',
        'first_sintering_temperature', 'first_sintering_time',
        'second_sintering_temperature', 'second_sintering_time'
    ]

    training_suffixes = ["mat_imputation", "knn_imputation", "mice_imputation", "mean_imputation"]
    

    shap.initjs()
    
    for suffix in training_suffixes:
        print(f"\nProcessing model interpretation for dataset '{suffix}' ({args.dataset_type})")
        model, scaler = load_model_and_scaler(suffix, args.dataset_type)
        if model is None or scaler is None:
            print(f"Skipping {suffix} because the model or scaler was not found.")
            continue
        
        X_df, y = load_training_data(suffix, args.dataset_dir, args.dataset_type, features)
        X = X_df.values 
        X_scaled = scaler.transform(X)
        
        plot_feature_importance(model, features, suffix, args.dataset_type)

        plot_shap_summary(model, X_scaled, features, suffix, args.dataset_type, param='loc')
        plot_shap_summary(model, X_scaled, features, suffix, args.dataset_type, param='scale')
   
        plot_shap_advanced(model, X_scaled, features, suffix, args.dataset_type, param='loc')
        plot_shap_advanced(model, X_scaled, features, suffix, args.dataset_type, param='scale')

        plot_shap_interaction(model, X_scaled, features, suffix, args.dataset_type, param='loc')
        plot_shap_interaction(model, X_scaled, features, suffix, args.dataset_type, param='scale')

if __name__ == "__main__":
    main()
