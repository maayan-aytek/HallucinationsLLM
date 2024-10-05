
from utils import *


WORD = 0
INDICES = 1
POS = 1

RELEVANT_COLUMNS = ['description', 'hallucinations', 'logits', 'words_logits_mapping', 'sentence', 'sentence_normalized_index', 'sentences_labels', 'sentences_preds', 
                    'sentence_contains_hedges', 'sentence_POS', 'sentence_probes']
SENTENCE_COLUMNS = ['sentence_normalized_index', 'sentences_labels', 'sentences_preds', 'sentence', 'sentence_contains_hedges', 'sentence_POS', 'sentence_len', 'sentence_probes']

stop_words_set = prepare_stopwords_set()
STOP_WORDS = (stop_words_set - {'above', 'all', 'any', 'are', "aren't", 'between', 'below', 'before', 'both', 'down', 'few',
                                    'each', 'from', 'further', 'he', 'her', 'here', 'hers', 'him', 'his', 'in', 'into',  'is', 'isn',
                                        "isn't", 'it', "it's", 'its', 'itself', 'most', 'no', 'not', 'on', 'of', 'off', 'only', 'once', 'other', 'over', 'out',
                                        'some', 'their', 'theirs', 'them', 'they', 'there', 'to', 'too', 'was', "wasn't", 'we', 'were', "weren't", ']', '['}).union(
                                    {'although', 'as', 'because', 'so', 'supposing', 'suggesting', 'additionally', 'that', 'than', 'though', 'till',
                                    'unless', 'until', 'when', 'whenever', 'accordingly', 'also', 'consequently', 'conversely', 'furthermore', 'finally', 'hence',
                                    'however', 'indeed', 'likewise', 'meanwhile', 'moreover', 'nevertheless', 'nonetheless', 'otherwise', 'similarly', 'therefore', 'thus'
                                    'whereas', 'wherever', 'whether', 'while', 'either', 'neither', 'but', 'yet'})

POS_HAL_PROBS = {'CD': 0.9038948454444954,
                'NNS': 0.7484338524421726,
                'JJR': 0.6597678493210688,
                'NN': 0.6120951444878695,
                'JJ': 0.6088321968881973,
                'PRP$': 0.560089986241773,
                'VBD': 0.5545456637290319,
                'PRP': 0.5416379491174973,
                'VBG': 0.5268154365979026,
                'RP': 0.48284981567558904,
                'NNP': 0.4492882230557565,
                'RBR': 0.4445346220533442,
                'CC': 0.34688308991201466,
                'VBP': 0.33743104365601634,
                'VB': 0.3355064018678578,
                'WDT': 0.3307895025804326,
                'RB': 0.32949244491597257,
                'VBN': 0.30182818962575186,
                'DT': 0.274220674558352,
                'IN': 0.26266726846556665,
                'TO': 0.25538859334929953,
                'VBZ': 0.16370153726126455,
                'EX': 0.11293980917389919,
                'MD': 0,
                'WRB': 0,
                'JJS': 0,
                'RBS': 0,
                'WP': 0,
                ':': 0}
