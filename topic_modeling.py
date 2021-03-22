#!/usr/bin/env python3
import os.path
from src.utils import cacheable
from src.tasks import task_list, prepare_task
from src.data.tokenizers import PretrainedTokenizer
import spacy
import tqdm
from gensim import corpora, models
import numpy as np
from argparse import ArgumentParser

from nltk.corpus import stopwords
import re
import string

puntuation_table = str.maketrans("", "", string.punctuation)


@cacheable(format="txt")
def prepare_data(
    task,
    tokenizer,
    dataset="train",
    spacy_model="en",
    remove_mentions=False,
    remove_urls=False,
    remove_punctuation=False,
    remove_stop_words=False,
    lang="english"
):
    if dataset in ["train", "valid", "test"]:
        dataset = getattr(task, f"{dataset}_data")

    out_data = []
    for x in dataset:
        sentence = tokenizer.decode(x.input_ids, True)
        out_data.append(sentence)

    # Filter out things before tokenizing
    def filter_fn(text):
        if remove_urls:
            text = remove_urls_from_text(text)
        if remove_mentions:
            text = remove_mentions_from_text(text)
        if remove_punctuation:
            text = remove_punctuation_from_text(text)
        return text

    print("Filtering data")
    filtered_data = [filter_fn(text) for text in out_data]
    # Tokenize
    print("Tokenizing data")
    tokenized_data = preprocess_text(filtered_data, spacy_model=spacy_model)

    if remove_stop_words:
        # Get stop words for the language
        stop_words = set(stopwords.words(lang))
        print("Removing stop words")
        tokenized_data = [
            " ".join([word for word in sent.split() if word not in stop_words])
            for sent in tokenized_data
        ]
    return tokenized_data


def remove_urls_from_text(tweet, replacement=""):
    """Remove urls from tweets
    from https://ourcodingclub.github.io/tutorials/topic-modelling-python/"""
    # remove http links
    tweet = re.sub(r"http\S+", replacement, tweet)
    # remove bitly links
    tweet = re.sub(r"bit.ly/\S+", replacement, tweet)
    # remove t.co links
    tweet = re.sub(r"t.co/\S+", replacement, tweet)
    tweet = tweet.strip("[link]")  # remove [links]
    return tweet


def remove_mentions_from_text(tweet, replacement=""):
    """Takes a string and removes retweet and @user information
    from https://ourcodingclub.github.io/tutorials/topic-modelling-python/"""
    # remove retweet
    tweet = re.sub(r"(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)", replacement, tweet)
    # remove tweeted at
    tweet = re.sub(r"(@[A-Za-z]+[A-Za-z0-9-_]+)", replacement, tweet)
    return tweet


def remove_punctuation_from_text(text):
    return text.translate(puntuation_table)


def preprocess_text(text, spacy_model="en"):
    if isinstance(spacy_model, str):
        nlp = spacy.load(spacy_model, disable=["tagger", "parser"])
    else:
        nlp = spacy_model
    tokenized_text = []
    for doc in tqdm.tqdm(nlp.pipe(text, n_threads=4, batch_size=500),
                         desc="Tokenizing"):
        tokenized_text.append(" ".join([token.text for token in doc]))

    return tokenized_text


def infer_topic(lda_model, bow):
    topics = lda_model.get_document_topics(bow)
    most_likely_topic, _ = max(topics, key=lambda x: x[1])
    return most_likely_topic


def get_args():
    parser = ArgumentParser("Train topic model on text data")
    # Experimental setting
    parser.add_argument("--random-seed", type=int, default=138413)
    parser.add_argument("--task", default="biased_SST_95",
                        choices=list(task_list.keys()),)
    parser.add_argument("--infer-only", action="store_true",
                        help="Just perform inference")
    parser.add_argument("--train-only", action="store_true",
                        help="Just perform training")
    # Model specific arguments
    parser.add_argument('--input-format', type=str, default=None,
                        choices=[None, "bert-base-uncased", "gpt2"],
                        help="Format (tok+vocabulary) for text input. "
                        "If None: decided based on the architecture.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite intermediary files")
    # LDA specific argument
    parser.add_argument("--remove-stop-words", action="store_true",
                        help="Remove stop words")
    parser.add_argument("--remove-punctuation", action="store_true",
                        help="Remove punctuation")
    parser.add_argument("--remove-urls", action="store_true",
                        help="Remove urls")
    parser.add_argument("--remove-mentions", action="store_true",
                        help="Remove twitter mentions")
    parser.add_argument("--lang", type=str, default="english",
                        help="Language (this is for determining stop words)")
    parser.add_argument("--n-topics", type=int, default=10)
    parser.add_argument("--max-vocab", type=int, default=10000)
    parser.add_argument("--n-inner-steps", type=int, default=2)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)

    # Task specific arguments
    return parser.parse_args()


def main():
    args = get_args()

    task, input_shape, output_size = prepare_task(
        args.task,
        model_name=args.input_format
    )

    tokenizer = PretrainedTokenizer(args.input_format)
    nlp = spacy.load("en", disable=["tagger", "parser"])
    # Dump training data
    cache_prefix = os.path.join("results", f"raw_{args.task}")
    train_data = prepare_data(
        task,
        tokenizer,
        dataset="train",
        spacy_model=nlp,
        cached_filename=f"{cache_prefix}_train.txt",
        overwrite=True,
        remove_mentions=args.remove_mentions,
        remove_urls=args.remove_urls,
        remove_punctuation=args.remove_punctuation,
        remove_stop_words=args.remove_stop_words,
        lang=args.lang,
    )
    print(len(train_data))
    valid_data = prepare_data(
        task,
        tokenizer,
        dataset="valid",
        spacy_model=nlp,
        cached_filename=f"{cache_prefix}_valid.txt",
        overwrite=True,
        remove_mentions=args.remove_mentions,
        remove_urls=args.remove_urls,
        remove_punctuation=args.remove_punctuation,
        remove_stop_words=args.remove_stop_words,
        lang=args.lang,
    )
    print(len(valid_data))
    test_data = prepare_data(
        task,
        tokenizer,
        dataset="test",
        spacy_model=nlp,
        cached_filename=f"{cache_prefix}_test.txt",
        overwrite=True,
        remove_mentions=args.remove_mentions,
        remove_urls=args.remove_urls,
        remove_punctuation=args.remove_punctuation,
        remove_stop_words=args.remove_stop_words,
        lang=args.lang,
    )
    print(len(test_data))
    # Split data
    train_data = [sent.split() for sent in train_data]
    valid_data = [sent.split() for sent in valid_data]
    test_data = [sent.split() for sent in test_data]
    # Creates, which is a mapping of word IDs to words.
    vocab = corpora.Dictionary(train_data, prune_at=args.max_vocab)

    # Turns each document into a bag of words.
    corpus = {
        "train": [vocab.doc2bow(sent) for sent in train_data],
        "valid": [vocab.doc2bow(sent) for sent in valid_data],
        "test": [vocab.doc2bow(sent) for sent in test_data],
    }

    # Topic model file name
    model_path = os.path.join("results", f"topic_model_{args.task}.gensim")
    # Training
    if not args.infer_only:
        # Train LDa model
        lda_model = models.ldamodel.LdaModel(
            corpus=corpus["train"],
            id2word=vocab,
            num_topics=args.n_topics,
            random_state=np.random.RandomState(args.random_seed),
            update_every=1,
            passes=args.n_iterations,
            iterations=args.n_inner_steps,
            alpha=args.alpha*np.ones(args.n_topics),
            eta=args.beta,
        )
        lda_model.save(model_path)
        for topic in lda_model.show_topics(args.n_topics, formatted=True):
            print(topic)
    # Inference
    if not args.train_only:
        # Inference
        lda_model = models.ldamodel.LdaModel.load(model_path)
        for split in ["train", "valid", "test"]:
            # Infer topics on this split
            topics = [infer_topic(lda_model, bow) for bow in corpus[split]]
            # Save as npy file
            filename = f"topics_{args.task}_{split}.npy"
            np.save(os.path.join("results", filename), topics)


if __name__ == "__main__":
    main()
