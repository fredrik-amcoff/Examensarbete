import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spacy
from spacy import displacy
from nltk import Tree
import pydot
from sklearn.preprocessing import StandardScaler
from scipy import stats


def sample_matched_pairs(df, n_pairs=1000):
    assert len(df) % 2 == 0, "Dataframe must be even"
    half = len(df) // 2

    indices = range(half)
    sampled_indices = pd.Series(indices).sample(n=n_pairs).values

    ai_samples = df.iloc[sampled_indices]
    human_samples = df.iloc[sampled_indices + half]
    matched_df = pd.concat([ai_samples, human_samples], ignore_index=True)
    return matched_df


def remove_outliers(df, variable, threshold=5):
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    df = df[(df[variable] >= lower_bound) & (df[variable] <= upper_bound)]
    return df


def generate_html_prompt(token_list, probs, top5, prob_threshold=0.7):
    """
    Generates the HTML for the figure under Theory - Summary of the transformer model - Transformers used for text
    generation. All parameters are created using generate_token_probs in data_generator.
    :param token_list: list of tokens
    :param probs: list of token probabilities
    :param top5: nested lists with pairs of token and probability:
    [[(token11, prob11), (token12, prob12)...], [(token_21, prob21)...]...]
    :param prob_threshold: threshold for including top 5 probs in HTML
    :return: None; creates HTML file
    """
    for i, tok in enumerate(token_list):
        token_list[i] = tok.replace(' ', '_')

    def prob_to_color(prob):
        red = int(255 * (1 - prob))
        green = int(255 * prob)
        blue = int(0)
        return f"rgb({red}, {green}, {blue})"

    html = """
    <div style='font-family: sans-serif; font-size: 16px;
                display: flex; flex-wrap: wrap;
                align-items: flex-start; gap: 2px;'>\n
    """

    for token, prob, top5_preds in zip(token_list, probs, top5):
        color = prob_to_color(prob)

        top5_html = ""
        if prob < prob_threshold:
            top5_html += "<div style='font-size: 10px; color: #666; line-height: 1.1; margin-top: 2px; text-align: left;'>"
            for alt_token, alt_prob in top5_preds:
                top5_html += f"{alt_token} ({alt_prob:.2f})<br>"
            top5_html += "</div>"

        html += f"""
        <div style="display: flex; flex-direction: column;
                    align-items: center; justify-content: flex-start;
                    padding: 2px; text-align: center; min-width: 1px;">
            <div style="color: {color}; font-weight: bold;">{token}</div>
            <div style="font-size: 11px; color: gray;">{prob:.2f}</div>
            {top5_html}
        </div>
        """

    html += "\n</div>"

    with open("token_probs_with_top5.html", "w", encoding="utf-8") as f:
        f.write(html)


def generate_text_len_graph():
    text_lens = []
    with open("translated_rnd.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for line in data:
            text_lens.append(line['wiki_words'])
    text_lens_2 = pd.read_csv('vital_articles_swe.csv')['length']

    plt.hist(text_lens_2, bins=10)
    plt.show()


def generate_data_hist(variable, file_name, bins=100, savepng=False, show=False, ax=None, iqr_threshold=5, xlabel=True, ylabel=True, legend=True, scale=False):
    df = pd.read_csv(file_name)
    df = remove_outliers(df, variable, iqr_threshold)
    data_min = df[variable].min()
    data_max = df[variable].max()
    bin_edges = np.linspace(data_min, data_max, bins + 1)

    ai = df[df['ai'] == 1]
    human = df[df['ai'] == 0]
    print(f'{var_names[variable]}:')
    print(f'AI mean: {round(np.mean(ai[variable].tolist()), 3)}')
    print(f'AI std: {round(np.std(ai[variable].tolist()), 3)}')
    print(f'Human mean: {round(np.mean(human[variable].tolist()), 3)}')
    print(f'Human std: {round(np.std(human[variable].tolist()), 3)}\n')


    if ax is None:
        plt.figure(figsize=(10,6))
        ax = plt.gca()

    ax.hist(ai[variable], bins=bin_edges, alpha=0.5, label='AI-generated', edgecolor='black', density=True)
    ax.hist(human[variable], bins=bin_edges, alpha=0.5, label='Human written', edgecolor='black', density=True)

    ymax = max(ax.get_ylim()[1], ax.get_ylim()[1])

    ax.vlines(np.mean(ai[variable].tolist()), ax.get_ylim()[0], ymax, linestyles='dashed', color='red', alpha=0.75)
    ax.vlines(np.mean(human[variable].tolist()), ax.get_ylim()[0], ymax, linestyles='dashed', color='red', alpha=0.75)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=10)
    if xlabel:
        ax.set_xlabel(var_names[variable], size=16)
    if ylabel:
        ax.set_ylabel('Density', size=16)
    if legend:
        ax.legend(fontsize=15)

    if savepng is True and ax is plt.gca():  # only save if standalone
        plt.savefig(f'Figures/{variable}_hist.png')
    if show is True and ax is plt.gca():
        plt.show()


def generate_data_scatter(variable, file_name, n=1000, savepng=False, show=False, ax=None):
    df = pd.read_csv(file_name)
    sample = sample_matched_pairs(df, n)

    ai = sample[sample['ai'] == 1]
    human = sample[sample['ai'] == 0]
    ai_mean = np.mean(ai[variable])
    human_mean = np.mean(human[variable])

    if ax is None:
        plt.figure(figsize=(10,6))
        ax = plt.gca()

    ax.scatter(ai[variable], human[variable], edgecolor='black')

    lower = max(ax.get_xlim()[0], ax.get_ylim()[0])
    upper = min(ax.get_xlim()[1], ax.get_ylim()[1])

    ax.vlines(ai_mean, ax.get_ylim()[0], ax.get_ylim()[1], linestyles='dashed', color='red', alpha=0.5)
    ax.hlines(human_mean, ax.get_xlim()[0], ax.get_xlim()[1], linestyles='dashed', color='red', alpha=0.5)
    ax.plot([lower, upper], [lower, upper], 'k--')

    ax.grid(True)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel(f'{var_names[variable]} (AI)', size=15)
    ax.set_ylabel(f'{var_names[variable]} (human)', size=15)

    if savepng is True and ax is plt.gca():
        plt.savefig(f'Figures/{variable}_scatter.png')
    if show is True and ax is plt.gca():
        plt.show()


def generate_data_kde(variable, file_name, savepng=False, show=False, ax=None, iqr_threshold=5, xlabel=True, ylabel=True, legend=True, bw_adjust=0.3):
    df = pd.read_csv(file_name)
    df = remove_outliers(df, variable, iqr_threshold)

    if ax is None:
        plt.figure(figsize=(10,6))
        ax = plt.gca()

    ai = df[df['ai'] == 1]
    human = df[df['ai'] == 0]

    sns.kdeplot(ai[variable].values, ax=ax, label='AI-written', color='blue', fill=True, alpha=0.5, bw_adjust=0.3)
    sns.kdeplot(human[variable].values, ax=ax, label='Human-written', color='orange', fill=True, alpha=0.5, bw_adjust=0.3)

    ymax = max(ax.get_ylim()[1], ax.get_ylim()[1])
    ax.vlines(np.mean(ai[variable].tolist()), ax.get_ylim()[0], ymax, linestyles='dashed', color='red', alpha=0.75)
    ax.vlines(np.mean(human[variable].tolist()), ax.get_ylim()[0], ymax, linestyles='dashed', color='red', alpha=0.75)

    ax.grid(True)
    ax.tick_params(axis='both', labelsize=10)
    if xlabel:
        ax.set_xlabel(var_names[variable], size=16)
    if ylabel:
        ax.set_ylabel('Density', size=16)
    if legend:
        ax.legend(fontsize=15)
    if savepng is True and ax is plt.gca():  # only save if standalone
        plt.savefig(f'Figures/{variable}_hist.png')
    if show is True and ax is plt.gca():
        plt.show()


def generate_data_subplots(variables, file_name, file_name_2, n=1000, bins=100, savepng=False, iqr_threshold=5, kde=False, bw_adjust=0.3):
    fig, axes = plt.subplots(2, len(variables), figsize=(5 * len(variables), 10))
    for col, variable in enumerate(variables):
        ylabel = False
        if col == 0:
            ylabel = True
        if kde:
            generate_data_kde(variable, file_name, ax=axes[0, col], iqr_threshold=iqr_threshold, xlabel=False, ylabel=ylabel, legend=False, bw_adjust=bw_adjust)
            generate_data_kde(variable, file_name_2, ax=axes[1, col], iqr_threshold=iqr_threshold, xlabel=True, ylabel=ylabel, legend=False, bw_adjust=bw_adjust)
        else:
            generate_data_hist(variable, file_name, ax=axes[0, col], bins=bins, iqr_threshold=iqr_threshold, xlabel=False, ylabel=ylabel, legend=False)
            generate_data_hist(variable, file_name_2, ax=axes[1, col], bins=bins, iqr_threshold=iqr_threshold, xlabel=True, ylabel=ylabel, legend=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=18)
    plt.tight_layout(rect=[0.05, 0.05, 1.0, 1.0])

    row_labels = ['English', 'Swedish']
    for row, label in enumerate(row_labels):
        fig.text(0.04, 0.75 - row * 0.42, label, ha='center', va='center', rotation='vertical', fontsize=28)

    # set equal xlims
    for col in range(len(variables)):
        xlims = [axes[row, col].get_xlim() for row in range(2)]
        print(xlims)
        xmin = min(x[0] for x in xlims)
        xmax = max(x[1] for x in xlims)
        for row in range(2):
            axes[row, col].set_xlim(xmin, xmax)
    if savepng is True and axes is not None:
        plt.savefig(f'Figures/{"-".join(variables)}-subplot_hist_only.png')
    plt.show()


def generate_dependency_tree(sentence):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    graph = pydot.Dot("dep_tree", graph_type="digraph", rankdir="TB", fontsize="10")

    # Add nodes
    for token in doc:
        graph.add_node(pydot.Node(
            str(token.i),
            label=f"{token.text}\n({token.dep_})",
            shape="plaintext"))

    # Add edges (dependencies)
    for token in doc:
        if token.head != token:
            graph.add_edge(pydot.Edge(str(token.head.i), str(token.i)))

    # Write to PNG
    output_path = "Figures/dep_tree.png"
    graph.write(output_path, format="png")


def calculate_ttests(file_name, variable):
    df = pd.read_csv(file_name)
    ai = df[df['ai'] == 1]
    human = df[df['ai'] == 0]
    ai_var = ai[variable].to_numpy()
    human_var = human[variable].to_numpy()
    t_stat, p_value = stats.ttest_rel(ai_var, human_var)
    diff = human_var - ai_var
    diff_mean = np.mean(diff)
    diff_std = np.std(diff, ddof=1)
    cohen_d = diff_mean/diff_std
    print(f'Mean of differences: {diff_mean:.2f}')
    print(f'Std.error: {diff_std:.2f}')
    print(f't-value: {t_stat:.2f}')
    print(f'p-value: {p_value:.3f}')
    print(f"Cohen's d: {cohen_d:.2f}")


def generate_pair_plots(file_path):
    df = pd.read_csv(file_path)
    df = sample_matched_pairs(df, 200)
    numerical_columns = [col for col in df.columns if col not in ['ai','title', 'topic', 'section', 'char_std',
                                                                  'word_std', 'temporal_burstiness',
                                                                  'syntactic_burstiness', 'wd_burstiness',
                                                                  'semantic_burstiness', 'words', 'chars']]
    sns.set_theme(style="whitegrid")

    # Create pairplot with category-based coloring
    g = sns.pairplot(df, vars=numerical_columns, hue="ai", palette={0: "orange", 1: "blue"}, plot_kws={'alpha': 0.4})
    output_path = 'pair_plot.png'
    g.savefig(output_path, dpi=300)
    plt.show()


def generate_heat_map(file_name):
    df = pd.read_csv(file_name)
    numerical_df = df[['perplexity', 'perplexity_std', 'intrinsic_dimensions', 'sentence_burstiness', 'word_burstiness',
                       'syntax_burstiness', 'unique_words', 'syntactic_depth', 'syntactic_repetitiveness', 'words',
                       'chars']]
    correlation_matrix = numerical_df.corr()
    print(correlation_matrix.to_string())
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.tight_layout()
    plt.show()




var_names = {'perplexity': 'Perplexity',
             'perplexity_std': 'Perplexity variability',
             'char_std': 'Sentence burstiness',
             'word_std': 'Sentence burstiness',
             'intrinsic_dimensions': 'Intrinsic dimensions',
             'sentence_burstiness': 'Sentence burstiness',
             'word_burstiness': 'Lemma burstiness',
             'syntax_burstiness': 'Syntactic burstiness',
             'unique_words': 'Unique words (%)',
             'syntactic_depth': 'Syntactic depth',
             'syntactic_repetitiveness': 'Syntactic repetitiveness',
             'syntactic_repetitiveness_2': 'Syntactic repetitiveness',
             'semantic_burstiness': 'Semantic burstiness (OLD)',
             'wd_burstiness': 'Word distribution burstiness (OLD)',
             'syntactic_burstiness': 'Syntactic burstiness (OLD)'}

var = ['perplexity', 'perplexity_std', 'intrinsic_dimensions']
generate_data_subplots(var, 'text_statistics_sv_eval.csv', savepng=True)
