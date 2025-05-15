import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_violin_probs_three_models(y_probs1, y_probs2, y_probs3, y_true, model_names=None, save_path="violin_probs_three_models_split.png"):
    """
    Plots violin plots for predicted probabilities from three models,
    arranged in subplots with Human on the top row and AI on the bottom,
    with large row labels on the left and x-axis labeled as density.

    Parameters:
        y_probs1, y_probs2, y_probs3 (array-like): Predicted probabilities from three models.
        y_true (array-like): True binary labels (0 for Human, 1 for AI).
        model_names (list of str, optional): Names for the models.
        save_path (str): File path to save the resulting plot.
    """
    if model_names is None:
        model_names = ['Logistic Regression', 'Random Forest', 'Neural Network']

    y_true = np.array(y_true)
    y_probs1 = np.array(y_probs1).flatten()
    y_probs2 = np.array(y_probs2).flatten()
    y_probs3 = np.array(y_probs3).flatten()

    model_probs = [y_probs1, y_probs2, y_probs3]

    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, (probs, name) in enumerate(zip(model_probs, model_names)):
        df = pd.DataFrame({
            'True Label': np.where(y_true == 0, 'Human', 'AI'),
            'Predicted Probability': probs
        })

        sns.violinplot(
            data=df[df['True Label'] == 'Human'],
            y='Predicted Probability',
            ax=axes[0, i],
            color=(0.6, 0.8, 1.0, 0.6),
            cut=0,
            inner=None,
            linewidth=1
        )
        axes[0, i].set_title(f"{name}", fontsize=22, weight='bold') 
        axes[0, i].set_xlabel("")  
        axes[0, i].set_ylim(0, 1)

        sns.violinplot(
            data=df[df['True Label'] == 'AI'],
            y='Predicted Probability',
            ax=axes[1, i],
            color=(1.0, 0.7, 0.8, 0.6),
            cut=0,
            inner=None,
            linewidth=1
        )
        axes[1, i].set_title("")
        axes[1, i].set_xlabel("Density") 
        axes[1, i].set_ylim(0, 1)

        if i == 0:
            axes[0, i].set_ylabel("Predicted Probability", fontsize=16)
            axes[1, i].set_ylabel("Predicted Probability", fontsize=16)
        else:
            axes[0, i].set_ylabel("")
            axes[1, i].set_ylabel("")
            axes[0, i].set_yticklabels([])
            axes[1, i].set_yticklabels([])

    fig.text(0.01, 0.76, "Human", va='center', ha='left', fontsize=22, weight='bold', rotation='vertical')
    fig.text(0.01, 0.26, "AI", va='center', ha='left', fontsize=22, weight='bold', rotation='vertical')

    plt.tight_layout(rect=[0.06, 0, 1, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()





true_values_eng = pd.read_csv("text_stats_eng_eval.csv")
true_values_eng = true_values_eng["ai"]

true_values_swe = pd.read_csv("text_stats_sv_eval.csv")
true_values_swe = true_values_swe["ai"]

predicted_probs_logistic = pd.read_csv("predictions_logistic.csv")
predicted_probs_forest = pd.read_csv("predictions_forest.csv")
predicted_probs_neural = pd.read_csv("predictions_neural.csv")


plot_violin_probs_three_models(predicted_probs_logistic, 
                               predicted_probs_forest, 
                               predicted_probs_neural, 
                               true_values_swe, 
                               model_names=None, 
                               save_path="violin_probs.png")