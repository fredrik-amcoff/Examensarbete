import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spacy
from spacy import displacy
from nltk import Tree
import pydot


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


def generate_data_hist(variable, file_name, bins=100, savepng=False, show=False, ax=None, iqr_threshold=5):
    df = pd.read_csv(file_name)
    df = remove_outliers(df, variable, iqr_threshold)
    data_min = df[variable].min()
    data_max = df[variable].max()
    bin_edges = np.linspace(data_min, data_max, bins + 1)

    ai = df[df['ai'] == 1]
    human = df[df['ai'] == 0]
    print(f'AI mean: {round(np.mean(ai[variable].tolist()), 3)}')
    print(f'AI std: {round(np.std(ai[variable].tolist()), 3)}')
    print(f'Human mean: {round(np.mean(human[variable].tolist()), 3)}')
    print(f'Human std: {round(np.std(human[variable].tolist()), 3)}')


    if ax is None:
        plt.figure(figsize=(10,6))
        ax = plt.gca()

    ax.hist(ai[variable], bins=bin_edges, alpha=0.5, label='AI-generated', edgecolor='black', density=True)
    ax.hist(human[variable], bins=bin_edges, alpha=0.5, label='Human written', edgecolor='black', density=True)

    ax.grid(True)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel(var_names[variable], size=15)
    ax.set_ylabel('Density', size=15)
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


def generate_data_subplots(variables, file_name, file_name_2, n=1000, bins=100, savepng=False, iqr_threshold=5):
    fig, axes = plt.subplots(2, len(variables), figsize=(5 * len(variables), 10))

    for col, variable in enumerate(variables):
        generate_data_hist(variable, file_name, ax=axes[0, col], bins=bins, iqr_threshold=iqr_threshold)
        generate_data_scatter(variable, file_name, ax=axes[1, col], n=n)

    plt.tight_layout()
    if savepng is True and axes is not None:
        plt.savefig(f'Figures/{"-".join(variables)}-subplot.png')
    plt.show()




var_names = {'perplexity': 'Perplexity',
             'perplexity_std': 'Perplexity variability',
             'char_std': 'Sentence burstiness',
             'word_std': 'Sentence burstiness',
             'intrinsic_dimensions': 'Intrinsic dimensions',
             'sentence_burstiness': 'Sentence burstiness',
             'word_burstiness': 'Lemma burstiness',
             'syntax_burstiness': 'Syntactic burstiness',
             'unique_words': 'Unique words',
             'syntactic_depth': 'Syntactic depth',
             'syntactic_repetitiveness': 'Syntactic repetitiveness',
             'semantic_burstiness': 'Semantic burstiness (OLD)',
             'wd_burstiness': 'Word distribution burstiness (OLD)',
             'syntactic_burstiness': 'Syntactic burstiness (OLD)'}

var = ['perplexity', 'perplexity_std', 'intrinsic_dimensions']
generate_data_subplots(var, 'text_statistics_eng_all.csv', savepng=True)