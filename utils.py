import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@dataclass
class PlotConfig:
    title_fontsize: 16
    label_fontsize: 14
    tick_fontsize: 14
    legend_fontsize: 12
    line_width: 2
    marker_size: int
    dpi_value: int


def plot_cv_parity(fold_data, y, suffix, avg_R2, config: PlotConfig, filename: str = None):
    plt.figure(figsize=(8, 8), dpi=config.dpi_value)
    colors = sns.color_palette("Dark2", len(fold_data))
    for idx, data in enumerate(fold_data):
        plt.scatter(data['y_true'], data['y_pred'], s=config.marker_size, alpha=0.7,
                    label=f'Fold {data["fold"]}', color=colors[idx])
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linewidth=config.line_width, label='Ideal')
    plt.xlabel('Actual Values', fontsize=config.label_fontsize)
    plt.ylabel('Predicted Mean', fontsize=config.label_fontsize)
    plt.title(f'CV Parity Plot - {suffix}', fontsize=config.title_fontsize)
    plt.legend(fontsize=config.legend_fontsize, loc='upper left')
    plt.xticks(fontsize=config.tick_fontsize)
    plt.yticks(fontsize=config.tick_fontsize)
    plt.annotate(f"Avg. R² = {avg_R2:.4f}", xy=(0.95, 0.95), xycoords='axes fraction',
                 fontsize=config.legend_fontsize, horizontalalignment="right",
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    
    if not filename:
        filename = f"plots/CV_parity_{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=config.dpi_value)
    plt.close()


def plot_cv_metrics_bar(folds, mse_scores, avg_MSE, mape_scores, avg_MAPE, 
                        r2_scores, avg_R2, nll_scores, avg_NLL, config: PlotConfig, filename: str = None):
    plt.figure(figsize=(12, 6), dpi=config.dpi_value)
 
    plt.subplot(2, 2, 1)
    plt.bar(folds, mse_scores, color='steelblue')
    plt.xlabel("Fold", fontsize=config.label_fontsize)
    plt.ylabel("MSE", fontsize=config.label_fontsize)
    plt.title("MSE per Fold", fontsize=config.title_fontsize)
    plt.xticks(fontsize=config.tick_fontsize)
    plt.yticks(fontsize=config.tick_fontsize)
    plt.text(0.95, 0.90, f"Avg. MSE = {avg_MSE:.4f}", transform=plt.gca().transAxes,
             fontsize=config.legend_fontsize, horizontalalignment="right",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    plt.subplot(2, 2, 2)
    plt.bar(folds, mape_scores, color='indianred')
    plt.xlabel("Fold", fontsize=config.label_fontsize)
    plt.ylabel("MAPE", fontsize=config.label_fontsize)
    plt.title("MAPE per Fold", fontsize=config.title_fontsize)
    plt.xticks(fontsize=config.tick_fontsize)
    plt.yticks(fontsize=config.tick_fontsize)
    plt.text(0.95, 0.90, f"Avg. MAPE = {avg_MAPE:.4f}", transform=plt.gca().transAxes,
             fontsize=config.legend_fontsize, horizontalalignment="right",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    
    plt.subplot(2, 2, 3)
    plt.bar(folds, r2_scores, color='seagreen')
    plt.xlabel("Fold", fontsize=config.label_fontsize)
    plt.ylabel("R² Score", fontsize=config.label_fontsize)
    plt.title("R² per Fold", fontsize=config.title_fontsize)
    plt.xticks(fontsize=config.tick_fontsize)
    plt.yticks(fontsize=config.tick_fontsize)
    plt.text(0.95, 0.90, f"Avg. R² = {avg_R2:.4f}", transform=plt.gca().transAxes,
             fontsize=config.legend_fontsize, horizontalalignment="right",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    

    plt.subplot(2, 2, 4)
    plt.bar(folds, nll_scores, color='mediumpurple')
    plt.xlabel("Fold", fontsize=config.label_fontsize)
    plt.ylabel("NLL", fontsize=config.label_fontsize)
    plt.title("NLL per Fold", fontsize=config.title_fontsize)
    plt.xticks(fontsize=config.tick_fontsize)
    plt.yticks(fontsize=config.tick_fontsize)
    plt.text(0.95, 0.90, f"Avg. NLL = {avg_NLL:.4f}", transform=plt.gca().transAxes,
             fontsize=config.legend_fontsize, horizontalalignment="right",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    if not filename:
        filename = "plots/CV_metrics_bar.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=config.dpi_value)
    plt.close()


def plot_loss_curves(fold_loss_history, final_iterations, final_train_losses, final_val_losses, config: PlotConfig, filename: str = None):
    plt.figure(figsize=(10, 6), dpi=config.dpi_value)
    palette = sns.color_palette("Dark2", len(fold_loss_history))
    for idx, fold_loss in enumerate(fold_loss_history):
        color = palette[idx % len(palette)]
        fold_num = fold_loss['fold']
        iters = fold_loss['iterations']
        t_losses = fold_loss['train_losses']
        v_losses = fold_loss['val_losses']
        if iters and t_losses and v_losses:
            plt.plot(iters, t_losses, marker='o', linestyle='-', color=color,
                     label=f'Fold {fold_num} Train Loss')
            plt.plot(iters, v_losses, marker='x', linestyle='--', color=color,
                     label=f'Fold {fold_num} Val Loss')
    if final_iterations and final_train_losses and final_val_losses:
        plt.plot(final_iterations, final_train_losses, marker='o', linestyle='-', color='black',
                 linewidth=config.line_width, label='Final Model Train Loss')
        plt.plot(final_iterations, final_val_losses, marker='x', linestyle='--', color='black',
                 linewidth=config.line_width, label='Final Model Val Loss')
    plt.xlabel('Iteration', fontsize=config.label_fontsize)
    plt.ylabel('Loss', fontsize=config.label_fontsize)
    plt.title('Iteration vs. Loss (Training & Validation) for Each Fold and Final Model', fontsize=config.title_fontsize)
    plt.legend(loc='upper right', fontsize=11, ncol=2, fancybox=True, framealpha=0.8)
    plt.xticks(fontsize=config.tick_fontsize)
    plt.yticks(fontsize=config.tick_fontsize)
    plt.tight_layout()
    if not filename:
        filename = "plots/loss_curves.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=config.dpi_value)
    plt.close()


def plot_uncertainty_heatmap(y_test_true, y_test_pred, predicted_std, metrics_annotation, config: PlotConfig, filename: str = None):
    plt.figure(figsize=(8, 8), dpi=config.dpi_value)
    sc = plt.scatter(
        y_test_true,
        y_test_pred,
        c=predicted_std,
        cmap='viridis',
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5,
        s=config.marker_size
    )
    cb = plt.colorbar(sc)
    cb.set_label('Predicted Std (Uncertainty)', fontsize=config.label_fontsize, rotation=90)
    plt.plot(
        [min(y_test_true), max(y_test_true)],
        [min(y_test_true), max(y_test_true)],
        'r--', linewidth=config.line_width, label='Ideal Fit'
    )
    plt.xlabel('Actual Values', fontsize=config.label_fontsize)
    plt.ylabel('Predicted Mean', fontsize=config.label_fontsize)
    plt.title('Heatmap (Test Set)', fontsize=config.title_fontsize)
    plt.text(
        0.02, 0.98,
        metrics_annotation,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
    )
    plt.subplots_adjust(right=0.8)
    if not filename:
        filename = "plots/uncertainty_heatmap.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=config.dpi_value)
    plt.close()


def plot_combined_calibration_curves(calibration_results, training_suffixes, config: PlotConfig, filename: str = None):
    plt.figure(figsize=(8, 6), dpi=config.dpi_value)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=config.line_width, label='Perfect Calibration')
    colors = sns.color_palette("Dark2", len(training_suffixes))
    for idx, suffix in enumerate(training_suffixes):
        quantiles, empirical_coverage, auce = calibration_results[suffix]
        plt.plot(quantiles, empirical_coverage, color=colors[idx], linewidth=config.line_width,
                 label=f'{suffix} (AUCE={auce:.4f})')
    plt.xlabel('Predicted Quantile', fontsize=config.label_fontsize)
    plt.ylabel('Empirical Coverage', fontsize=config.label_fontsize)
    plt.title('Combined Calibration Curves Comparison (Test Set)', fontsize=config.title_fontsize)
    plt.legend(fontsize=config.legend_fontsize, loc='best')
    plt.xticks(fontsize=config.tick_fontsize)
    plt.yticks(fontsize=config.tick_fontsize)
    if not filename:
        filename = "plots/combined_calibration_curves.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=config.dpi_value)
    plt.close()


def plot_fold_parity_with_uncertainty(fold_dict, suffix, config: PlotConfig, filename: str = None):
    """
    Generate a parity scatter plot for an individual fold where points are color-coded by their predicted uncertainty.
    """
    fold = fold_dict['fold']
    y_true = fold_dict['y_true']
    y_pred = fold_dict['y_pred']
    y_std = fold_dict['y_std']
    plt.figure(figsize=(8, 8), dpi=config.dpi_value)
    sc = plt.scatter(y_true, y_pred, c=y_std, cmap='viridis', s=config.marker_size, alpha=0.7, edgecolor='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=config.line_width, label='Ideal')
    plt.xlabel('Actual Values', fontsize=config.label_fontsize)
    plt.ylabel('Predicted Mean', fontsize=config.label_fontsize)
    plt.title(f'Parity Plot with Uncertainty (Fold {fold} - {suffix})', fontsize=config.title_fontsize)
    plt.colorbar(sc, label='Predicted Std (Uncertainty)')
    plt.legend(fontsize=config.legend_fontsize)
    if not filename:
         filename = f"plots/Parity_with_uncertainty_fold{fold}_{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=config.dpi_value)
    plt.close()


def plot_final_parity_with_uncertainty(y_true, y_pred, y_std, suffix, config: PlotConfig, filename: str = None):
    """
    Generate a parity plot for the final trained model (applied on the training set),
    with points color-coded by the predicted uncertainty.
    """
    plt.figure(figsize=(8, 8), dpi=config.dpi_value)
    sc = plt.scatter(y_true, y_pred, c=y_std, cmap='viridis', s=config.marker_size, alpha=0.7, edgecolor='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=config.line_width, label='Ideal')
    plt.xlabel('Actual Values', fontsize=config.label_fontsize)
    plt.ylabel('Predicted Mean', fontsize=config.label_fontsize)
    plt.title(f'Final Model Parity Plot with Uncertainty ({suffix})', fontsize=config.title_fontsize)
    plt.colorbar(sc, label='Predicted Std (Uncertainty)')
    plt.legend(fontsize=config.legend_fontsize)
    if not filename:
        filename = f"plots/Final_parity_with_uncertainty_{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=config.dpi_value)
    plt.close()
