import os
import pandas as pd
import spacy
from typing import Union, Sequence, Literal

import torch
from torch.utils.data import Dataset


nlp = spacy.load('en_core_web_sm')


class Vocabulary:
    def __init__(self, frequency_threshold: Union[int, float] = 0):
        """
        Initializes the Vocabulary object with a frequency threshold.

        Args:
            frequency_threshold (Union[int, float]): The frequency threshold for including words in the vocabulary.
                If an integer is provided, it represents the minimum number of occurrences a word must have to be included.
                If a float is provided, it must be between 0 and 1, representing the minimum frequency proportion a word must have to be included.
        """
        self.index2str = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.str2index = {v: k for k, v in self.index2str.items()}
        self.freq_threshold = frequency_threshold

    def __len__(self):
        return len(self.index2str)

    @staticmethod
    def tokenize(text: str):
        """
        Tokenizes the input text using the spaCy tokenizer.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            List[str]: A list of tokens extracted from the input text, converted to lowercase.
        """
        return [token.text.lower() for token in nlp(text)]

    def build_vocab(self, texts: Sequence[str]):
        """
        Builds the vocabulary from a sequence of texts.

        This method processes a list of input texts, tokenizes each text, and counts the frequency of each token.
        Based on the specified frequency threshold, it filters out tokens that do not meet the threshold criteria.
        The remaining tokens are then added to the vocabulary, with each token assigned a unique index.

        Args:
            texts (Sequence[str]): A sequence of input texts to build the vocabulary from.

        Returns:
            None
        """
        token_cnt = {}
        for text in texts:
            for token in self.tokenize(text):
                if token not in token_cnt:
                    token_cnt[token] = 1
                else:
                    token_cnt[token] += 1

        if isinstance(self.freq_threshold, int):
            tokens = [k for k, v in token_cnt.items() if v > self.freq_threshold]
        else:
            total_cnt = sum([v for k, v in token_cnt.imtes()])
            tokens = [
                k for k, v in token_cnt.items() if v / total_cnt >= self.freq_threshold
            ]

        for idx, token in enumerate(tokens):
            self.index2str[idx + 4] = token
            self.str2index[token] = idx + 4

    def numericalize(self, text: str):
        """
        Converts the input text into a list of numerical indices based on the vocabulary.

        This method tokenizes the input text and maps each token to its corresponding numerical index
        using the vocabulary built by the `build_vocab` method. If a token is not found in the vocabulary,
        it is replaced with the index for the "<UNK>" token.

        Args:
            text (str): The input text to be converted into numerical indices.

        Returns:
            List[int]: A list of numerical indices representing the tokens in the input text.
        """
        tokens = self.tokenize(text)

        return [
            self.str2index[t] if t in self.str2index else self.str2index["<UNK>"]
            for t in tokens
        ]


class AGNewsDataset(Dataset):
    """
    AGNewsDataset is a custom Dataset class for loading and processing the AG News dataset.

    The dataset file must be in .csv format and should be located in the specified data directory.
    The .csv file should contain the following columns:

    - 'Class Index': The category label of the news article (e.g., World, Sports, Business, Sci/Tech).
    - 'Title': The title of the news article.
    - 'Description': The description or content of the news article.

    The class supports loading both training and testing splits of the dataset.
    """

    _LABEL = "Class Index"
    _TITLE = "Title"
    _DESCRIPTION = "Description"


    def __init__(
        self,
        data_dir: str,
        get_title: bool = False,
        max_sentece_len: int = 100,
        max_document_len: int = 40,
        split: Literal["train", "test"] = "train",
    ):
        """
        Initializes the AGNewsDataset object by loading and processing the dataset from the specified directory.

        Args:
            data_dir (str): The directory where the dataset .csv file is located.
            get_title (bool): If True, the title of the news article will be concatenated with the description. Default is False.
            max_sentece_len (int): The maximum words of a sentence. Default is 100.
            max_document_len (int): The maximum sentences of a document. Default is 40.
            split (Literal['train', 'test']): The dataset split to load, either 'train' or 'test'. Default is 'train'.
        """
        super().__init__()

        # get data
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))

        self.labels = df.loc[:, self._LABEL].tolist()

        if get_title:
            df[self._DESCRIPTION] = df[self._TITLE] + "\n" + df[self._DESCRIPTION]
        self.texts = df.loc[:, self._DESCRIPTION].tolist()

        # create vocab and tokenizer
        self.vocab = Vocabulary()
        self.vocab.build_vocab(self.texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # get label
        label = self.labels[idx]

        # get setences and words
        tokenized_documents = torch.tensor(
            [self.vocab.numericalize(sen.text) for sen in nlp(self.texts[idx]).sents]

        )

        return label, tokenized_documents