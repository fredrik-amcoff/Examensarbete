import torch as t
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
from IntrinsicDimCUDA import PHD
import re


def clean_text(text):
    # Remove references
    text = re.sub(r"\[\d+\]|\(\d{4}\)", "", text)
    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9\s.,!?()'\"-]", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove newlines
    text = text.replace('\n', ' ').replace('  ', ' ')
    return text


def split_text_sliding_window(prompt, window_size=50, step_size=25):
    """Splits text into overlapping windows."""
    words = prompt.split()
    return [" ".join(words[i:i+window_size]) for i in range(0, len(words)-window_size+1, step_size)]


def split_text_into_chunks(text, chunk_size=50):
    """Splits text into chunks of approximately chunk_size words."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def test_prompt_return(
    prompt: str,
    answer: Union[str, list[str]],
    model,
    prepend_bos: Optional[bool] = None,):
    """
    Calculates the probability distribution of next word prediction given a context
    :param prompt: represents the context
    :param answer: actual next word
    :param model: transformer model
    :param prepend_bos:
    :return:
    """

    answers = [answer] if isinstance(answer, str) else answer
    n_answers = len(answers)
    using_multiple_answers = n_answers > 1

    # GPT-2 often treats the first token weirdly, so lets give it a resting position
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    answer_tokens = model.to_tokens(answers, prepend_bos=False)

    # If we have multiple answers, we're only allowed a single token generation
    if using_multiple_answers:
        answer_tokens = answer_tokens[:, :1]

    # Deal with case where answers is a list of strings
    prompt_tokens = prompt_tokens.repeat(answer_tokens.shape[0], 1)
    tokens = t.cat((prompt_tokens, answer_tokens), dim=1)

    # Turns the text and answer into tokens
    prompt_str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
    answer_str_tokens_list = [model.to_str_tokens(answer, prepend_bos=False) for answer in answers]
    prompt_length = len(prompt_str_tokens)
    answer_length = 1 if using_multiple_answers else len(answer_str_tokens_list[0])

    # Logit and probability of each token in the vocabulary
    logits = model(tokens)
    probs = logits.softmax(dim=-1)
    answer_ranks = []
    token_probs = probs[:, prompt_length - 1]
    logit_result = logits[0, prompt_length-1, answer_tokens[0]].item()
    prob_result = token_probs[0, answer_tokens[0]].item()
    return logit_result, prob_result


def get_perplexity_data(prompt, model, prnt=True):
    prompt_str_tokens = model.to_str_tokens(prompt)
    prompt_len = len(prompt_str_tokens)
    token_str = ""
    prob_list = []
    odds_list = []
    token_list = []
    logit_list = []
    times = []
    # attention_len = 150

    # iterate through tokens
    for i, token in enumerate(prompt_str_tokens):
        start_time = time.time()
        prob_results = test_prompt_return(token_str, token, model)
        token_str += token
        if prnt == True:
            print(f'{i}/{prompt_len}')
            print(token)
            print(prob_results)
        # print(np.log(prob_results[1]/(1-prob_results[1])))
        prob_list.append(prob_results[1])
        odds_list.append(np.log(prob_results[1] / (1 - prob_results[1])))
        token_list.append(token)
        logit_list.append(prob_results[0])
        end_time = time.time()
        times.append(end_time - start_time)
    return {
        'prob_list': prob_list,
        'odds_list': odds_list,
        'token_list': token_list,
        'times': times
    }


def get_perplexity(prompt, model, tokenizer, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with t.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    #logits = outputs.logits
    perplexity = t.exp(loss).item()

    #probs = t.nn.functional.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    # Extract probabilities of actual tokens
    #token_probs = probs[0, :-1, :]  # Ignore the last token since it has no next-token prediction
    #actual_token_probs = token_probs.gather(1, input_ids[:, 1:].T)  # Get the prob of the actual next token

    # Convert to a readable format
    #decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    #for token, prob in zip(decoded_tokens[:-1], actual_token_probs.squeeze().tolist()):
    #    print(f"Token: {token.ljust(10)} | Probability: {prob:.6f}")
    return perplexity


def get_perplexity_variability(prompt, model, tokenizer, device, chunk_size=50, chunk_type="standard", language="english"):
    if chunk_type == 'sliding_window':
        chunks = split_text_sliding_window(prompt, chunk_size)
    elif chunk_type == 'sentence':
        chunks = sent_tokenize(prompt, language=language)
    else:
        chunks = split_text_into_chunks(prompt, chunk_size)
    if len(chunks) < 2:
        return 0  # Not enough segments to compute variance
    ppls = []

    for chunk in chunks:
        chars = len(chunk.strip())
        if chars > 0:
            ppl = get_perplexity(chunk, model, tokenizer, device)
            ppls.append(ppl)

    return np.std(ppls)


def get_sentence_burstiness(prompt):
    # Split the text into sentences
    sentences = prompt.split('.')
    char_lengths = []
    word_lengths = []
    for sentence in sentences:
        chars = len(sentence.strip())
        if chars > 0:
            words = len(sentence.split())
            char_lengths.append(chars)
            word_lengths.append(words)

    char_std = np.std(char_lengths)
    char_var = np.var(char_lengths)
    word_std = np.std(word_lengths)
    word_var = np.var(word_lengths)

    return {
        'char_std': char_std,
        'char_var': char_var,
        'word_std': word_std,
        'word_var': word_var
    }


def get_temporal_burstiness(prompt):
    """
    Returns the Fano factor
    """
    words = prompt.split()
    word_counts = Counter(words)
    occurrences = np.array(list(word_counts.values()))
    return np.var(occurrences) / np.mean(occurrences)


def get_syntactic_burstiness(prompt, nlp):
    doc = nlp(prompt)
    pos_sequences = [token.pos_ for token in doc]
    pos_counts = Counter(pos_sequences)
    occurrences = np.array(list(pos_counts.values()))
    return np.var(occurrences) / np.mean(occurrences)


def get_wd_burstiness(prompt, chunk_size=50, chunk_type='standard'):
    """Computes TF-IDF burstiness by measuring variance in word distributions across chunks."""
    if chunk_type == 'sliding_window':
        chunks = split_text_sliding_window(prompt, chunk_size)
    else:
        chunks = split_text_into_chunks(prompt, chunk_size)

    if len(chunks) < 2:
        return 0  # Not enough segments to compute variance

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks).toarray()

    word_variances = np.var(tfidf_matrix, axis=0)  # Variance across segments for each word
    return np.mean(word_variances)  # Average variance across all words


def get_lda_burstiness(prompt, num_topics=3, chunk_size=50, chunk_type='standard'):
    """Computes semantic burstiness by analyzing topic variability across text chunks."""

    # Tokenize and segment text
    if chunk_type == 'sliding_window':
        chunks = split_text_sliding_window(prompt, chunk_size)
    else:
        chunks = split_text_into_chunks(prompt, chunk_size)

    tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]

    # Create a dictionary and corpus for LDA
    dictionary = Dictionary(tokenized_chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in tokenized_chunks]

    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Get topic distributions per chunk
    topic_distributions = np.array([
        [prob for _, prob in lda_model.get_document_topics(doc, minimum_probability=0)]
        for doc in corpus
    ])

    # Compute variance across topic distributions
    topic_variance = np.var(topic_distributions, axis=0)

    return np.mean(topic_variance, dtype=np.float64)  # Higher variance = more burstiness


def get_intrinsic_dimensions(prompt, model, tokenizer, device, min_subsample=40, intermediate_points=7):
    """
    This function is a modified version from https://github.com/ArGintum/GPTID
    :param prompt: input text
    :param model: input model
    :param tokenizer: input tokenizer
    :return:
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) #PUT ONTO GPU FOR (CUDA)
    with t.no_grad():
        outp = model(**inputs).logits[0].cpu() #PUT BACK ONTO CPU (CUDA)
    # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
    mx_points = inputs['input_ids'].shape[1] - 2

    mn_points = min_subsample
    step = (mx_points - mn_points) // intermediate_points
    return PHD().fit_transform(outp.numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step,
                               point_jump=step)



def get_statistics(text_data, model, tokenizer, device, nlp, chunk_size=50, chunk_type='standard', num_topics=3):
    """
    Get all statistics for several texts (from json file).
    :param text_data: json file with text data
    :param model: GPT-model
    :param tokenizer: GPT-tokenizer
    :param device: example cpu or cuda
    :param nlp: for syntactic burstiness
    :param chunk_size: number of tokens
    :param chunk_type: "standard" or "sliding_window"
    :param num_topics: number of topics in lda burstiness
    :return: None, stores data in two csv files
    """
    def collect_statistics(text, label):
        sb_data = get_sentence_burstiness(text)
        return {
            "perplexity": get_perplexity(text, model, tokenizer, device),
            "perplexity_std": get_perplexity_variability(text, model, tokenizer, device, chunk_type=chunk_type),
            "char_std": sb_data["char_std"],
            "word_std": sb_data["word_std"],
            "intrinsic_dimensions": get_intrinsic_dimensions(text, model, tokenizer, device),
            "temporal_burstiness": get_temporal_burstiness(text),
            "syntactic_burstiness": get_syntactic_burstiness(text, nlp),
            "wd_burstiness": get_wd_burstiness(text, chunk_size=chunk_size, chunk_type=chunk_type),
            "semantic_burstiness": get_lda_burstiness(text, num_topics=num_topics, chunk_size=chunk_size, chunk_type=chunk_type),
            "ai": label
        }

    measures = ["perplexity", "perplexity_std", "char_std", "word_std", "intrinsic_dimensions", "temporal_burstiness",
                "syntactic_burstiness", "wd_burstiness", "semantic_burstiness", "ai"]

    statistics_ai = {key: [] for key in measures}
    statistics_human = {key: [] for key in measures}

    times = []
    with open(text_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    index = 0
    for entry in data:
        start_time = time.time()
        index += 1
        # Print information
        print(f'\nArticle {index}: {entry["title"]}')
        print(f'{entry["ai_words"]} words (AI)')
        print(f'{entry["wiki_words"]} words (wiki)')
        human_text = clean_text(entry["wikipedia_text"])
        ai_text = clean_text(entry["ai_text"])

        ai_stats = collect_statistics(ai_text, label=1)
        human_stats = collect_statistics(human_text, label=0)

        # Append to dataset
        for key in statistics_ai.keys():
            statistics_ai[key].append(ai_stats[key])
            statistics_human[key].append(human_stats[key])

        stop_time = time.time()
        print(f"Time: {stop_time - start_time}")
        times.append(stop_time - start_time)
    df_ai = pd.DataFrame(statistics_ai)
    df_human = pd.DataFrame(statistics_human)
    df_combined = pd.concat([df_ai, df_human], ignore_index=True)
    df_combined.to_csv('text_statistics.csv', index=False, encoding='utf-8')

    print(f'Times: {times}')
    print(f'Total time: {sum(times)}')


device_1 = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


nltk.download('punkt_tab')  # used for lda burstiness
nlp_1 = spacy.load("en_core_web_sm")  # used for syntactic burstiness
model_3 = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")
tokenizer_3 = AutoTokenizer.from_pretrained("ai-forever/mGPT")

model_3.to(device_1)

get_statistics("translated_rnd.json", model_3, tokenizer_3, device_1, nlp_1, chunk_type='sliding_window')