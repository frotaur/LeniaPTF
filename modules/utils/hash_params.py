import os
import pickle
import nltk

import torch
import hashlib
from typing import Dict, Set
from nltk.corpus import wordnet as wn


def fetch_nouns_and_adj_from_nltk():
    """
    this func is a one time call to download a database of nouns and adjectives to be stored as pickle files,
    the point is to avoid the database being changed in time since the nltk online version could be updated.

    Should be useless, as we provide the files in the repo
    """
    adj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adjectives.pk')
    noun_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nouns.pk')
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        raise Exception(f"Error downloading NLTK data: {str(e)}")

    adjectives: Set[str] = set()
    nouns: Set[str] = set()

    # Get adjectives
    for synset in list(wn.all_synsets(wn.ADJ)):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if word.isalpha() and len(word) > 2:  # Filter out very short words and non-alphabetic
                adjectives.add(word)
    
    # Get nouns
    for synset in list(wn.all_synsets(wn.NOUN)):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if word.isalpha() and len(word) > 2:  # Filter out very short words and non-alphabetic
                nouns.add(word)

    # Convert sets to sorted lists for deterministic behavior
    adjectives = sorted(list(adjectives))
    nouns = sorted(list(nouns))

    #save to files
    with open(adj_path, 'wb') as handle:
        pickle.dump(adjectives, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(noun_path, 'wb') as handle:
        pickle.dump(nouns, handle, protocol=pickle.HIGHEST_PROTOCOL)

def params_to_words(state_dict: Dict[str, torch.Tensor], num_words: int = 2) -> str:
    """
    Convert neural network state dictionary into a deterministic sequence of words.
    Acts like a hash function - similar parameters will generate the same words.
    
    Args:
        state_dict: Model state dictionary loaded from torch.load()
        num_words: Number of words to generate
    
    Returns:
        String containing space-separated words
    """
    adj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adjectives.pk')
    noun_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nouns.pk')
    # Load adjectives and nouns
    with open(adj_path, 'rb') as handle:
        adjectives = pickle.load(handle)
    with open(noun_path, 'rb') as handle:
        nouns = pickle.load(handle)

    # Convert state dict to bytes for hashing
    param_bytes = b''
    for key in sorted(state_dict.keys()):  # Sort keys for deterministic ordering
        if not(isinstance(state_dict[key],torch.Tensor)):
            tohash = torch.tensor(state_dict[key])
        else:
            tohash = state_dict[key]
        # use torch.tensor as a hack, to convert the 'int' of k_size to bytes
        param_bytes += tohash.cpu().numpy().tobytes() 
    
    # Create deterministic seed from parameters
    hash_value = int(hashlib.sha256(param_bytes).hexdigest(), 16)
    
    # Generate words using the hash value
    words = []
    for i in range(num_words):
        seed = (hash_value + i * 12345) & 0xFFFFFFFF
        word_list = adjectives if i % 2 == 0 else nouns
        word_idx = seed % len(word_list)
        words.append(word_list[word_idx])
    
    return "_".join(words)
