import torch as t
import torch_directml
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import HookedTransformer
import time
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer



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
    print(2)
    with t.no_grad():
        outputs = model(input_ids, labels=input_ids)
        print(3)
        loss = outputs.loss
        logits = outputs.logits
        perplexity = t.exp(loss).item()
        print(4)
    probs = t.nn.functional.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    print(5)
    # Extract probabilities of actual tokens
    token_probs = probs[0, :-1, :]  # Ignore the last token since it has no next-token prediction
    actual_token_probs = token_probs.gather(1, input_ids[:, 1:].T)  # Get the prob of the actual next token

    # Convert to a readable format
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for token, prob in zip(decoded_tokens[:-1], actual_token_probs.squeeze().tolist()):
        print(f"Token: {token.ljust(10)} | Probability: {prob:.6f}")
    return perplexity


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
    statistics_ai = {
        "perplexity": [],
        "char_std": [],
        "word_std": [],
        "temporal_burstiness": [],
        "syntactic_burstiness": [],
        "wd_burstiness": [],
        "semantic_burstiness": [],
        "ai": []
    }
    statistics_human = {
        "perplexity": [],
        "char_std": [],
        "word_std": [],
        "temporal_burstiness": [],
        "syntactic_burstiness": [],
        "wd_burstiness": [],
        "semantic_burstiness": [],
        "ai": []
    }
    times = []
    with open(text_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data[0:1]:
        start_time = time.time()
        title = entry["title"]
        print(f"Article: {title}")
        print(f'{entry["ai_words"]} words (AI)')
        print(f'{entry["wiki_words"]} words (wiki)')
        wikipedia_text = entry["wikipedia_text"]
        ai_text = entry["ai_text"]
        ai_sb_data = get_sentence_burstiness(ai_text)
        hum_sb_data = get_sentence_burstiness(wikipedia_text)
        # Append to ai dataset
        statistics_ai["perplexity"].append(get_perplexity(ai_text, model, tokenizer, device))
        statistics_ai["char_std"].append(ai_sb_data["char_std"])
        statistics_ai["word_std"].append(ai_sb_data["word_std"])
        statistics_ai["temporal_burstiness"].append(get_temporal_burstiness(ai_text))
        statistics_ai["syntactic_burstiness"].append(get_syntactic_burstiness(ai_text, nlp))
        statistics_ai["wd_burstiness"].append(get_wd_burstiness(ai_text, chunk_size=chunk_size, chunk_type=chunk_type))
        statistics_ai["semantic_burstiness"].append(
            get_lda_burstiness(ai_text, num_topics=num_topics, chunk_size=chunk_size, chunk_type=chunk_type))
        statistics_ai["ai"].append(1)
        # Append to human dataset
        statistics_human["perplexity"].append(get_perplexity(wikipedia_text, model, tokenizer, device))
        statistics_human["char_std"].append(hum_sb_data["char_std"])
        statistics_human["word_std"].append(hum_sb_data["word_std"])
        statistics_human["temporal_burstiness"].append(get_temporal_burstiness(wikipedia_text))
        statistics_human["syntactic_burstiness"].append(get_syntactic_burstiness(wikipedia_text, nlp))
        statistics_human["wd_burstiness"].append(get_wd_burstiness(wikipedia_text, chunk_size=chunk_size, chunk_type=chunk_type))
        statistics_human["semantic_burstiness"].append(
            get_lda_burstiness(wikipedia_text, num_topics=num_topics, chunk_size=chunk_size, chunk_type=chunk_type))
        statistics_human["ai"].append(0)
        stop_time = time.time()
        print(f"Time: {stop_time - start_time}")
        times.append(stop_time - start_time)
    df_ai = pd.DataFrame(statistics_ai)
    df_ai.to_csv('statistics_ai.csv', index=False, encoding='utf-8')
    df_human = pd.DataFrame(statistics_human)
    df_human.to_csv('statistics_human.csv', index=False, encoding='utf-8')
    print(statistics_human)
    print(f'Times: {times}')
    print(f'Total time: {sum(times)}')


device_1 = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
#device_2 = torch_directml.device()


#nltk.download('punkt_tab')  # used for lda burstiness
#nlp_1 = spacy.load("en_core_web_sm")  # used for syntactic burstiness
#model_1 = GPT2LMHeadModel.from_pretrained("gpt2")
#tokenizer_1 = GPT2Tokenizer.from_pretrained("gpt2")
# BLOOM
#model_2 = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
#tokenizer_2 = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
# mGPT
model_3 = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")
tokenizer_3 = AutoTokenizer.from_pretrained("ai-forever/mGPT")
#model_4 = HookedTransformer.from_pretrained("gpt2-medium", device = device_1)

model_3.to(device_1)
print(1)
time_1 = time.time()
prompt_1 = """
Europa (från grekiskans: Ευρώπη) är jordens näst minsta världsdel till ytan men tredje folkrikaste, med lite mer än 750 miljoner invånare (2023) varav över 90 procent talar språk som tillhör den indoeuropeiska språkfamiljen. Europa är världens näst mest tätbefolkade världsdel och har den näst högsta produktiviteten per person. Världsdelen Europa utgör den västligaste delen av kontinenten Eurasien, och avgränsas av Atlanten i väster (gränsen mot Nordamerika går genom Danmarksundet mellan Island och det danska autonoma landet Grönland), Medelhavet i söder och Norra ishavet i norr, medan gränsen mot Asien i öster går längs Uralbergen, Uralfloden, Kaspiska havet, Kaukasus vattendelare och Svarta havet (som är ett innanhav av Medelhavet).

De första moderna människorna kom till Europa för cirka 40 000 år sedan.[13] Under istiden var de nordligaste delarna obeboeliga och befolkningen levde i refugier i södra och sydöstra Europa, efter istiden expanderade de europeiska stammarna norrut. Under yngre stenåldern och bronsåldern skedde stora kulturella och ekonomiska omvandlingar, oftast med centrum kring Medelhavet. Under Romarrikets storhetstid under de första seklerna e.Kr. växte den nya religionen kristendomen snabbt. Under renässansen grundlades den dynamik som gav upphov till den europeiska upplysningen, den industriella revolutionen och vetenskapens framväxt.[14] Under andra hälften av det andra millenniet e.Kr. skedde snabb befolkningstillväxt och miljontals européer utvandrade till andra världsdelar. Främst Frankrike, Storbritannien, Portugal, Nederländerna och Spanien, men även flera andra länder, skaffade sig kolonier och utländska besittningar och spred europeisk kultur globalt.[15]

Allt färre européer är under 2000-talet troende kristna,[16] och majoriteten av yngre människor beräknas under 2000-talet sakna religiös tillhörighet.[17] Europas politik präglas i dag av samarbetet inom Europeiska unionen (EU), som växte fram ur tidigare europeiska samarbeten och i dag består av 27 medlemsstater (varav 26 ligger geografiskt i Europa, medan Cypern ligger geografiskt i Asien). Östeuropa präglas geografiskt, ekonomiskt och politiskt av ett stort Ryssland, vilket under större delen av 1900-talet fungerade som den centrala delen i Sovjetunionen.
"""

tokens = tokenizer_3(prompt_1, return_tensors="pt")

num_tokens = tokens.input_ids.shape[1]
print(num_tokens)

print(get_perplexity(prompt_1, model_3, tokenizer_3, device_1))

#pp_data = get_perplexity_data(prompt_1, model_1)
#print(pp_data)
#
time_2 = time.time()
print(f'Run time: \n{time_2 - time_1}')
#print(f'Prob mean: \n{np.mean(pp_data["prob_list"])}')
#print(f'Prob std: \n{np.std(pp_data["prob_list"])}')
#print(f'Odds mean: \n{np.mean(pp_data["odds_list"])}')
#print(f'Odds std: \n{np.std(pp_data["odds_list"])}')
#print(f'Mean run time per token: \n{np.mean(pp_data["times"])}')
#
#plt.plot(pp_data[3])
#plt.show()
#plt.hist(pp_data[0], bins=100, density=True)
#plt.show()
#plt.hist(pp_data[1], bins=50, density=True)
#plt.show()


"""
df = pd.read_csv("logodds_hum.csv")
plt.hist(df["logit"], bins=100, density=True, alpha=0.5)
df_gen = pd.read_csv("logodds_gen.csv")
plt.hist(df_gen["logit"], bins=100, density=True, alpha=0.5)
plt.show()
print(np.mean(df["logit"]))
print(np.mean(df_gen["logit"]))
"""
#new_data = {
#    "token": token_list,
#    "logit": logit_list,
#    "prob": prob_list,
#    "logodds": odds_list
#}
#
#new_df = pd.DataFrame(new_data)
#df = pd.concat([df, new_df], ignore_index=True)
#df.to_csv("logodds_hum.csv", index=False)



#test_prompt(prompt, answer, model, prepend_space_to_answer=False)