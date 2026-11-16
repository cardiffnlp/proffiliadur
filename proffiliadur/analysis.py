"""
This file:

- performs readability analysis by:
  - calling on pos_tagger.py to get POS counts using CyTag2
  - calling on syntax_analyser.py to run UDPipe server request

- computes the Welsh and English readability/linguistic feature sets
  consumed by app.py's /api/* routes

"""

import os
import os.path
import json
import logging
import math
from functools import lru_cache
import regex as re
import numpy as np
import pandas as pd
import nltk

logger = logging.getLogger(__name__)

# NLTK data is expected to be pre-provisioned by setup.sh, so a server
# restart never depends on network access. Fails fast with a clear message
# rather than silently downloading (which would mask a deployment that
# isn't actually persisting this data between restarts).
for _resource_path in ('tokenizers/punkt_tab', 'corpora/stopwords', 'taggers/averaged_perceptron_tagger_eng'):
    try:
        nltk.data.find(_resource_path)
    except LookupError:
        raise RuntimeError(
            f"Missing NLTK resource '{_resource_path}'. Run backend/setup.sh to provision it."
        )

from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

""" Other custom files from project, needed for grammar/syntax analysis """
from .pos_tagger import POSTagger
from .syntax_analyser import SyntaxAnalyser
# SyntaxAnalyser has no mutable instance state written during analyse(), so
# it's safe to share across concurrent requests. POSTagger is not (see
# readability_analysis).
synt = SyntaxAnalyser()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


""" Helper Functions: """

def G(cp, pf): # Matricciani's formulas (see https://doi.org/10.3390/analytics2020016 for more info)
  res = 89 - (11.1)*cp + (300/pf) if pf != 0 else 0
  return res

def Gu(G, Ip):
  res = G - 6*(Ip-6)
  return res

def Gu2(Ip):
  res = -7.74*Ip +103.12
  return res

def get_cp(text, verbose=False):
  words = re.findall(r'\b\w+\b', text)  # Matches words (ignores punctuation)
  if verbose: logger.debug(f"Words: {words}")
  total_chars = sum(len(word) for word in words)
  total_words = len(words)
  return total_chars/total_words if total_words != 0 else 0

def get_pf(text, verbose=False):
  sentences = re.split(r'[.!?]+(?:\s|$)', text.strip())
  sentences = [s for s in sentences if s]  # Remove empty strings
  if verbose: logger.debug(f"Sentences: {sentences}")
  words = re.findall(r'\b\w+\b', text)
  total_words = len(words)
  total_sents = len(sentences)
  return total_words/total_sents if total_sents != 0 else 0

def get_ip(text, verbose=False):
  # Split on all interpunction marks (commas, semicolons, dashes, etc.)
  segments = re.split(r'[.,;:!?()\[\]{}"\-–—]', text)
  segments = [seg.strip() for seg in segments if seg.strip()]
  if verbose: logger.debug(f"Interpunction segments: {segments}")
  word_counts = [len(re.findall(r'\b\w+\b', seg)) for seg in segments]
  total_words = sum(word_counts)
  total_separators = len(word_counts)
  return total_words / total_separators if total_separators != 0 else 0

def calculate(text, verbose=False):
  if isinstance(text, list):
    text = ' '.join(text)
  cp = get_cp(text) if not verbose else get_cp(text, verbose)
  pf = get_pf(text) if not verbose else get_pf(text, verbose)
  ip = get_ip(text) if not verbose else get_ip(text, verbose)
  if verbose: logger.debug(f"CP: {cp:0,.3f}, PF: {pf:.3f}, IP: {ip:.3f}")
  if pf == 0:
    logger.warning("Cannot compute Gulpease score: pf (words per sentence) is 0")
    g = 0
  else:
    g = G(cp, pf)
  gu = Gu(g, ip)
  gu2 = Gu2(ip)
  if verbose: logger.debug(f"G = {g:.3f}\n Gu = {gu:.3f}\n Gu2 = {gu2:.3f}")
  return g, gu, gu2

def sentence_count(txt):
        sentences = re.split(r'[.!?]+(?:\s|$)', txt.strip())
        sentences = [s for s in sentences if s]  # Remove empty strings
        return len(sentences)

def deduplicate_wordforms(instances_list):
    """helper for unique wordform instances """
    seen = set()
    unique_instances = []
    for row in instances_list:
        wordform = row['wordform']
        if wordform not in seen:
            seen.add(wordform)
            unique_instances.append(row)
    return unique_instances

def readability_analysis(text):
    # WELSH readability analysis

    # Setting up files for readability analysis
    vocab_files = {
        "a1": os.path.join(BASE_DIR, "data", "vocab", "a1_geirfa.json"),
        "a2": os.path.join(BASE_DIR, "data", "vocab", "a2_geirfa.json"),
        "b1": os.path.join(BASE_DIR, "data", "vocab", "b1_geirfa.json"),
    }
    stopwords_files = {
        "cy_ataleiriau": os.path.join(BASE_DIR, "data", "stopwords", "cy_ataleiriau.txt"),
    }

    #inits
    if "\\n" in text:
      text = text.replace("\\n", " ")


    # A fresh POSTagger instance per call keeps its counters isolated to
    # this request - POSTagger.__init__ does no I/O, so this is cheap.
    local_pos = POSTagger()
    df, results_cefr, results_band, results_stopwords, counters = local_pos.analyse_vocab(text, vocab_files, stopwords_files=stopwords_files)
    results = {}
    instances = {}

    #--------------------Surface Level Statistics--------------------

    sent_count = sentence_count(text)

    results["Sentence Count"] = sent_count

    """ All other tokenisation and lemmatisation is done through CyTag, so uses the counters from pos class: """
    total_tokens = sum(counters['tokens'].values())
    results["Token Count"] = total_tokens
    clean_text = re.sub(r'[\s_.,;:!?"\'\(\)]+', '', text)
    results["Mean Word Length"] = len(clean_text) / sum(counters["words"].values()) if counters["words"] else 0
    results["Number Count"] = sum(counters["numbers"].values())
    results["Word Count"] = sum(counters["words"].values())

    results["Average Sentence Length"] = sum(counters["words"].values()) / sent_count if sent_count else 0
    results["Unique Words"] = len(counters["words"]) if counters["words"] else 0
    results["TTR (Type Token Ratio)"] = len(counters["words"])/sum(counters["words"].values()) if counters["words"] else 0
    results["TTR (Token Type Ratio)"] = sum(counters["words"].values())/len(counters["words"]) if counters["words"] else 0


    #--------------------Vocabulary Lists--------------------


    #df, results_cefr, results_band, results_stopwords = pos.analyse_vocab(text, vocab_files, stopwords_files=stopwords_files)

    for level, result in results_cefr.items(): #CEFR lists
        results[f"{level} - unique matches"] = result["unique_matches"]
        results[f"{level} - total appearances"] = result["total_appearances"]
        results[f"{level} - ratio to all words"] = result["total_appearances"]/total_tokens if total_tokens else 0


    for rank, amount in results_band.items(): #CorCenCC lists
        results[f"{rank} - unique matches"] = amount["unique_matches"]
        results[f"{rank} - total appearances"] = amount["total_appearances"]
        results[f"{rank} - ratio to all words"] = amount["total_appearances"]/total_tokens if total_tokens else 0


    # --- Stopword statistics ---
    for stopword_type, result in results_stopwords.items():
        base = f"{stopword_type}"

        results[f"{base} - unique stopwords"] = result.get("unique_stopwords", 0)
        results[f"{base} - total appearances"] = result.get("total_appearances", 0)
        results[f"{base} - ratio to all words"] = (
            result["total_appearances"] / total_tokens if total_tokens else 0
        )



    instances["Wordlist instances"] = df.to_dict(orient="records")
    unique_wordform_instances = deduplicate_wordforms(instances["Wordlist instances"])
    instances["Unique Wordlist instances"] = unique_wordform_instances

    # CEFR instances (level-based)
    cefr_matches = df[df["level"].notna()]
    instances["CEFR instances"] = [
        {"wordform": row["wordform"], "lemma": row["lemma"], "category": row["level"]}
        for _, row in cefr_matches.iterrows()
    ]

    # Frequency band instances
    band_matches = df[df["band"].notna()]
    instances["Frequency band instances"] = [
        {"wordform": row["wordform"], "lemma": row["lemma"], "category": row["band"]}
        for _, row in band_matches.iterrows()
    ]

    # Stopword instances (can be a list of categories per word)
    stopword_cols = [c for c in df.columns if c.startswith("stopword_")]
    stopword_matches = df[df[stopword_cols].any(axis=1)]

    instances["Stopword instances"] = []

    for _, row in stopword_matches.iterrows():
      # find which stopword columns matched for this token
      matched_cats = [col.replace("stopword_", "") for col in stopword_cols if row[col]]

      for cat in matched_cats:
          instances["Stopword instances"].append({
              "wordform": row["wordform"],
              "lemma": row["lemma"],
              "category": cat
          })

    #--------------------Lexical Diversity--------------------
    # results["MTLD"] = mtld
    # results["HDD"] = hdd
    # print(f"MTLD: {results['MTLD']:.4}")
    # print(f"HDD: {results['HDD']:.4}")

    #--------------------PoS Tagging--------------------

    results["Coverage of POS tags"] = sum(counters["tags"].values())

    results["Number of types of POS Tags"] = len(counters["tags"].values())
    results["Lemma Count"] = len(counters["lemma"])

    # VERBS
    verb_count = counters["verb"]
    results["Number of verbs"] = sum(verb_count.values())
    results["Unique verb lemmas"] = len(counters["verb-lemma"])
    results["Average verb count per word"] = sum(verb_count.values()) / total_tokens if total_tokens else 0

    results["Number of verbs in Present tense"] = sum(counters["pres-v"].values())
    results["Ratio of verbs in Present tense to all verbs"] = sum(counters["pres-v"].values())/ sum(verb_count.values()) if verb_count else 0

    results["Number of verbs in Past tense"] = sum(counters["gorff-v"].values())
    results["Ratio of verbs in Past tense to all verbs"] = sum(counters["gorff-v"].values())/ sum(verb_count.values()) if verb_count else 0

    results["Number of verbs in Imperfect tense"] = sum(counters["amherff-v"].values())
    results["Ratio of verbs in Imperfect tense to all verbs"] = sum(counters["amherff-v"].values())/ sum(verb_count.values()) if verb_count else 0

    results["Number of verbs in Future tense"] = sum(counters["dyf-v"].values())
    results["Ratio of verbs in Future tense to all verbs"] = sum(counters["dyf-v"].values())/ sum(verb_count.values()) if verb_count else 0

    results["Number of verbs in Imperative tense"] = sum(counters["gorch-v"].values())
    results["Ratio of verbs in Imperative tense to all verbs"] = sum(counters["gorch-v"].values())/ sum(verb_count.values()) if verb_count else 0

    results["Number of verbs in Conditional tense"] = sum(counters["amod-v"].values())
    results["Ratio of verbs in Conditional tense to all verbs"] = sum(counters["amod-v"].values())/ sum(verb_count.values()) if verb_count else 0

    results["Number of verbs in Subjunctive tense"] = sum(counters["dibdyf-v"] .values())
    results["Ratio of verbs in Subjunctive tense to all verbs"] = sum(counters["dibdyf-v"] .values())/ sum(verb_count.values()) if verb_count else 0

    results["Number of verb nouns"] = sum(counters["e-v"].values())
    results["Ratio of verb nouns to all verbs"] = sum(counters["e-v"].values())/ sum(verb_count.values()) if verb_count else 0

    results["Number of verbs in Pluperfect tense"] = sum(counters["gorb-v"].values())
    results["Ratio of verbs in Pluperfect tense to all verbs"] = sum(counters["gorb-v"].values()) / sum(verb_count.values()) if verb_count else 0

    results["Number of verbs in Impersonal tense"] = sum(counters["amhers-v"].values())
    results["Ratio of verbs in Impersonal tense to all verbs"] = sum(counters["amhers-v"].values()) / sum(verb_count.values()) if verb_count else 0

    results["Number of Perfect tenses"] = sum(counters["perf-v"].values())
    results["Average perfect tenses per word"] = sum(counters["perf-v"].values()) / total_tokens if total_tokens else 0

    # NOUNS
    noun_count = counters["noun"]
    results["Number of Nouns"] = sum(noun_count.values())
    results["Average nouns per word"] = sum(noun_count.values()) / total_tokens if total_tokens else 0
    results["Unique noun lemmas"] = len(counters["noun-lemma"])

    results["Number of Masculine nouns"] = sum(counters["masc-n"].values())
    results["Ratio of Masculine nouns to all nouns"] = sum(counters["masc-n"].values())/ sum(noun_count.values()) if noun_count else 0

    results["Number of Feminine nouns"] = sum(counters["fem-n"].values())
    results["Ratio of Feminine nouns to all nouns"] = sum(counters["fem-n"].values())/ sum(noun_count.values()) if noun_count else 0

    results["Number of Singular nouns"] = sum(counters["sing-n"].values())
    results["Ratio of Singular nouns to all nouns"] = sum(counters["sing-n"].values())/ sum(noun_count.values()) if noun_count else 0

    results["Number of Plural nouns"] = sum(counters["pl-n"].values())
    results["Ratio of Plural nouns to all nouns"] = sum(counters["pl-n"].values())/ sum(noun_count.values()) if noun_count else 0

    results["Number of Proper nouns"] = sum(counters["proper-n"].values())
    results["Ratio of Proper nouns to all nouns"] = sum(counters["proper-n"].values())/ sum(noun_count.values()) if noun_count else 0

    #EBYCH
    results["Number of expressions"] = sum(counters["ebych"].values())
    results["Average expressions per word"] = sum(counters["ebych"].values()) / total_tokens if total_tokens else 0

    #ADJ
    results["Number of adjectives"] = sum(counters["adj"].values())
    results["Average adjectives per word"] = sum(counters["adj"].values()) / total_tokens if total_tokens else 0
    results["Number of plural adjectives"] = sum(counters["plural-adj"].values())
    results["Average plural adjectives per word"] = sum(counters["plural-adj"].values()) / total_tokens if total_tokens else 0
    #ADV
    results["Number of adverbs"] = sum(counters["adv"].values())
    results["Average adverbs per word"] = sum(counters["adv"].values()) / total_tokens if total_tokens else 0
    #prepositions
    results["Number of prepositions"] = sum(counters["prep"].values())
    results["Average prepositions per word"] = sum(counters["prep"].values()) / total_tokens if total_tokens else 0
    results["Number of uninflected prepositions"] = sum(counters["prep-uni"].values())
    results["Ratio of uninflected prepositions to all prepositions"] = sum(counters["prep-uni"].values()) / sum(counters["prep"].values()) if counters["prep"] else 0
    results["Number of inflected prepositions"] = sum(counters["prep-in"].values())
    results["Ratio of inflected prepositions to all prepositions"] = sum(counters["prep-in"].values()) / sum(counters["prep"].values()) if counters["prep"] else 0
    #numerals
    results["Number of numerals"] = sum(counters["numeral"].values())
    results["Average numerals per word"] = sum(counters["numeral"].values()) / total_tokens if total_tokens else 0
    #articles
    results["Number of articles"] = sum(counters["article"].values())
    results["Average articles per word"] = sum(counters["article"].values()) / total_tokens if total_tokens else 0
    #punctuations
    results["Number of punctuations"] = sum(counters["punctuation"].values())
    results["Average punctuations per word"] = sum(counters["punctuation"].values()) / total_tokens if total_tokens else 0

    #conjunctions

    results["Number of conjunctions"] = sum(counters["conj"].values())
    results["Average conjunctions per word"] = sum(counters["conj"].values()) / total_tokens if total_tokens else 0

    results["Number of Coordinating conjunctions"] = sum(counters["conj-coord"].values())
    results["Ratio of Coordinating conjunctions to all Conjunctions"] = sum(counters["conj-coord"].values()) / sum(counters["conj"].values()) if counters["conj"] else 0
    results["Number of Subordinating conjunctions"] = sum(counters["conj-sub"].values())
    results["Ratio of Subordinating conjunctions to all Conjunctions"] = sum(counters["conj-sub"].values()) / sum(counters["conj"].values()) if counters["conj"] else 0

    #pronouns
    results["Number of pronouns"] = sum(counters["pronoun"].values())
    results["Average pronouns per word"] = sum(counters["pronoun"].values()) / total_tokens if total_tokens else 0

    results["Number of personal pronouns"] = sum(counters["rhapers"].values())
    results["Ratio of personal pronouns to all pronouns"] = sum(counters["rhapers"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of dependent pronouns"] = sum(counters["rhadib"].values())
    results["Ratio of dependent pronouns to all pronouns"] = sum(counters["rhadib"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of possessive pronouns"] = sum(counters["rhamedd"].values())
    results["Ratio of possessive pronouns to all pronouns"] = sum(counters["rhamedd"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of conjunctive pronouns"] = sum(counters["rhacys"].values())
    results["Ratio of conjunctive pronouns to all pronouns"] = sum(counters["rhacys"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of interrogative pronouns"] = sum(counters["rhagof"].values())
    results["Ratio of interrogative pronouns to all pronouns"] = sum(counters["rhagof"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of demonstrative pronouns"] = sum(counters["rhadang"].values())
    results["Ratio of demonstrative pronouns to all pronouns"] = sum(counters["rhadang"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of relative pronouns"] = sum(counters["rhaperth"].values())
    results["Ratio of relative pronouns to all pronouns"] = sum(counters["rhaperth"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of reflexive pronouns"] = sum(counters["rhaatb"].values())
    results["Ratio of reflexive pronouns to all pronouns"] = sum(counters["rhaatb"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of reciprocal pronouns"] = sum(counters["rhacil"].values())
    results["Ratio of reciprocal pronouns to all pronouns"] = sum(counters["rhacil"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0
    results["Number of impersonal pronouns"] = sum(counters["rhaamh"].values())
    results["Ratio of impersonal pronouns to all pronouns"] = sum(counters["rhaamh"].values()) / sum(counters["pronoun"].values()) if counters["pronoun"] else 0

    #determiners Ban
    results["Number of determiners"] = sum(counters["determiner"].values())
    results["Average determiners per word"] = sum(counters["determiner"].values()) / total_tokens if total_tokens else 0
    results["Number of simple-early determiners"] = sum(counters["bancynnar"].values())
    results["Ratio of simple-early determiners to all determiners"] = sum(counters["bancynnar"].values()) / sum(counters["determiner"].values()) if counters['determiner'] else 0
    results["Number of simple-late determiners"] = sum(counters["banhwyr"].values())
    results["Ratio of simple-late determiners to all determiners"] = sum(counters["banhwyr"].values()) / sum(counters["determiner"].values()) if counters['determiner'] else 0
    results["Number of demonstrative determiners"] = sum(counters["bandang"].values())
    results["Ratio of demonstrative determiners to all determiners"] = sum(counters["bandang"].values()) / sum(counters["determiner"].values()) if counters['determiner'] else 0
    results["Number of interrogative determiners"] = sum(counters["bangof"].values())
    results["Ratio of interrogative determiners to all determiners"] = sum(counters["bangof"].values()) / sum(counters["determiner"].values()) if counters["determiner"] else 0
    results["Number of possessive determiners"] = sum(counters["banmedd"].values())
    results["Ratio of possessive determiners to all determiners"] = sum(counters["banmedd"].values()) / sum(counters["determiner"].values()) if counters["determiner"] else 0
    results["Number of quantifier determiners"] = sum(counters["banmeint"].values())
    results["Ratio of quantifier determiners to all determiners"] = sum(counters["banmeint"].values()) / sum(counters["determiner"].values()) if counters["determiner"] else 0

    #particles
    results["Number of particles"] = sum(counters["particle"].values())
    results["Average particles per word"] = sum(counters["particle"].values()) / total_tokens if total_tokens else 0


    #NEW register markers:
    demonstratives_count = sum(counters["bandang"].values()) + sum(counters["rhadang"].values())
    results["Number of demonstratives"] = demonstratives_count
    results["Average demonstratives per word"] = demonstratives_count / total_tokens if total_tokens else 0

    subordinates_count = sum(counters["rhaperth"].values()) + sum(counters["perth-v"].values())
    results["Number of subordinates"] = subordinates_count
    results["Average subordinates per word"] = subordinates_count / total_tokens if total_tokens else 0

    results["Number of second person"] = sum(counters["second-p"].values())
    results["Average number of second person per word"] =  sum(counters["second-p"].values()) / total_tokens if total_tokens else 0

    results["Number of third person"] = sum(counters["third-p"].values())
    results["Average number of third person per word"] =  sum(counters["third-p"].values()) / total_tokens if total_tokens else 0

    results["Number of nominalisations"] = sum(counters["nomin"].values())
    results["Average nominalisations per word"] = sum(counters["nomin"].values()) / total_tokens if total_tokens else 0

    results["Number of modals"] = sum(counters["modals"].values())
    results["Average modals per word"] = sum(counters["modals"].values()) / total_tokens if total_tokens else 0

    results["Number of conjuncts"] = sum(counters["conjuncts"].values())
    results["Average conjuncts per word"] = sum(counters["conjuncts"].values()) / total_tokens if total_tokens else 0
    results["Number of downtoners"] = sum(counters["downtoners"].values())
    results["Average downtoners per word"] = sum(counters["downtoners"].values()) / total_tokens if total_tokens else 0
    results["Number of hedges"] = sum(counters["hedges"].values())
    results["Average hedges per word"] = sum(counters["hedges"].values()) / total_tokens if total_tokens else 0
    results["Number of amplifiers"] = sum(counters["amplifiers"].values())
    results["Average amplifiers per word"] = sum(counters["amplifiers"].values()) / total_tokens if total_tokens else 0
    results["Number of emphatics"] = sum(counters["emphatics"].values())
    results["Average emphatics per word"] = sum(counters["emphatics"].values()) / total_tokens if total_tokens else 0
    results["Number of discourse particles"] = sum(counters["discourse-parts"].values())
    results["Average discourse particles per word"] = sum(counters["discourse-parts"].values()) / total_tokens if total_tokens else 0

    results["POS counts"] = {
      "verb": results["Number of verbs"],
      "noun": results["Number of Nouns"],
      "particle": results["Number of particles"],
      "article": results["Number of articles"],
      "pronoun": results["Number of pronouns"],
      "preposition": results["Number of prepositions"],
      "conjunction": results["Number of conjunctions"],
      "adjective": results["Number of adjectives"],
      "adverb": results["Number of adverbs"],
      "numeral": results["Number of numerals"],
      "determiner": results["Number of determiners"]
    }

    # -- Mutation Counts --

    results["Number of mutations"] = sum(counters["muts"].values())
    results["Average number of mutations per word"] = sum(counters["muts"].values()) / total_tokens if total_tokens else 0
    results["Number of Soft mutations"] = sum(counters["sm"].values())
    results["Average number of soft mutations per word"] = sum(counters["sm"].values()) / total_tokens if total_tokens else 0
    results["Number of Nasal mutations"] = sum(counters["nm"].values())
    results["Average number of nasal mutations per word"] = sum(counters["nm"].values()) / total_tokens if total_tokens else 0
    results["Number of Aspirate mutations"] = sum(counters["am"].values())
    results["Average number of aspirate mutations per word"] = sum(counters["am"].values()) / total_tokens if total_tokens else 0
    results["Number of h-prothesis"] = sum(counters["hm"].values())
    results["Average number of h-prothesis per word"] = sum(counters["hm"].values()) / total_tokens if total_tokens else 0

    # ------- Readability Formulas (adapted) -----

    g1, g2, g3 = calculate(text)
    results["Adapted Gulpease readability 1"] = g1
    results["Adapted Gulpease readability 2"] = g2
    results["Adapted Gulpease readability 3"] = g3

    # -- Syntactic Features --
    syn_feats = synt.analyse(text, "cy")
    results["Max Tree depth"] = syn_feats["max_tree_height"] if syn_feats["max_tree_height"] else 0
    results["Max Tree breadth"] = syn_feats["max_tree_breadth"] if syn_feats["max_tree_breadth"] else 0
    results["Average depth"] = syn_feats["avg_depth"] if syn_feats["avg_depth"] else 0
    results["Average breadth"] = syn_feats["avg_breadth"] if syn_feats["avg_breadth"] else 0
    results["Max branching factor"] = syn_feats["max_branching_factor"] if syn_feats["max_branching_factor"] else 0
    results["Average head distance"] = syn_feats["avg_head_distance"] if syn_feats["avg_head_distance"] else 0
    results["Subject ratio"] = syn_feats["subject_ratio"] if syn_feats["subject_ratio"] else 0
    results["Object ratio"] = syn_feats["object_ratio"] if syn_feats["object_ratio"] else 0
    results["Average subject depth"] = np.mean(syn_feats["subject_depths"]) if syn_feats["subject_depths"] else 0
    results["Average object depth"] = np.mean(syn_feats["object_depths"]) if syn_feats["object_depths"] else 0
    results["Average verb depth"] = syn_feats["avg_verb_depth"] if syn_feats["avg_verb_depth"] else 0
    results["Average noun depth"] = syn_feats["avg_noun_depth"] if syn_feats["avg_noun_depth"] else 0
    results["Left-right branching ratio"] = syn_feats["left_right_branching_ratio"] if syn_feats["left_right_branching_ratio"] else 0
    results["Average subject-head distance"] = syn_feats["avg_subj_head_distance"] if syn_feats["avg_subj_head_distance"] else 0
    results["Average object-head distance"] = syn_feats["avg_obj_head_distance"] if syn_feats["avg_obj_head_distance"] else 0
    results["Total clause count"] = syn_feats["clause_count"] if syn_feats["clause_count"] else 0
    results["Clause count per sentence"] = syn_feats["avg_clause_count_per_sent"] if syn_feats["avg_clause_count_per_sent"] else 0
    results["Clause Density (clause per tokens)"] = syn_feats["clause_density"] if syn_feats["clause_density"] else 0
    results["Number of coordinated conjuncts"] = syn_feats["conj_count"] if syn_feats["conj_count"] else 0
    results["Number of auxiliaries"] = syn_feats["aux_count"] if syn_feats["aux_count"] else 0
    results["Average leaf ratio"] = syn_feats["avg_leaf_ratio"] if syn_feats["avg_leaf_ratio"] else 0


    # get the words

    instances["Verb instances"] = list(counters["ins_verbs"])
    instances["Noun instances"] = list(counters["ins_nouns"])
    instances["Full instance"] = list(counters["full_instance"])
    instances["Other instances"] = list(counters["ins_others"])
    instances["Adjective instances"] = list(counters["ins_adjectives"])
    instances["Adverb instances"] = list(counters["ins_adverbs"])
    instances["Preposition instances"] = list(counters["ins_prepositions"])
    instances["Particle instances"] = list(counters["ins_particles"])
    instances["Article instances"] = list(counters["ins_articles"])
    instances["Conjunction instances"] = list(counters["ins_conjunctions"])
    instances["Pronoun instances"] = list(counters["ins_pronouns"])
    instances["Determiner instances"] = list(counters["ins_determiners"])
    instances["Numeral instances"] = list(counters["ins_numerals"])

    return results, instances


def readability_analysis_english(text):

    results = {}

    #surface level first
    sents = sent_tokenize(text)
    results['Sentence Count'] = len(sents)
    tokens = nltk.word_tokenize(text)
    results['Token Count'] = len(tokens)
    results['Number Count'] = len([token for token in tokens if token.isnumeric()])
    results['Word Count'] = len([token for token in tokens if token.isalpha()])
    results['Average Sentence Length'] = len(tokens) / len(sents)
    results['Unique Words'] = len(set(tokens))
    results['TTR (Type Token Ratio)'] = len(set(tokens)) / len(tokens)
    results['TTR (Token Type Ratio)'] = len(tokens) / len(set(tokens))

    # wordlists
    cefr_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'vocab', 'english_cerf_words.csv'))
    freq_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'bnc_frequency_k_bands.csv'))

    # CEFR
    levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    tokens_lower = [token.lower() for token in tokens]
    total_tokens = len(tokens)

    for level in levels:
        cefr_set = set(cefr_df.loc[cefr_df['CEFR'] == level, 'headword'].str.lower())

        matches = [t for t in tokens_lower if t in cefr_set]

        results[f'{level} - total appearances'] = len(matches)
        results[f'{level} - unique matches'] = len(set(matches))
        results[f'{level} - ratio to all words'] = len(matches) / total_tokens if total_tokens > 0 else 0

    #freq
    bands = ['K1', 'K2', 'K3', 'K4', 'K5', 'K6+']

    for band in bands:
        freq_set = set(freq_df.loc[freq_df['k_band'] == band, 'word'].str.lower())

        matches = [t for t in tokens_lower if t in freq_set]

        results[f'{band} - total appearances'] = len(matches)
        results[f'{band} - unique matches'] = len(set(matches))
        results[f'{band} - ratio to all words'] = len(matches) / total_tokens if total_tokens > 0 else 0

    #stopwords
    stops = set(stopwords.words('english'))
    results['unique stopwords'] = len(set([token for token in tokens if token.lower() in stops]))
    results['total stopwords'] = len([token for token in tokens if token.lower() in stops])
    results['stopwords - ratio to all words'] = results['total stopwords'] / len(tokens)

    # pos tags

    tags = pos_tag(tokens)

    nouns = [word for word, pos in tags if pos.startswith('NN')]

    results['Number of Nouns'] = len(nouns)
    results['Average nouns per word'] = len(nouns) / len(tokens) if len(tokens) > 0 else 0



    pl_nouns = [word for word, pos in tags if pos in ('NNS', 'NNPS')]

    results["Number of Plural nouns"] = len(pl_nouns) if pl_nouns else 0
    results["Ratio of Plural nouns to all nouns"] = len(pl_nouns)/ len(nouns) if pl_nouns else 0

    sing_nouns = [word for word, pos in tags if pos in ('NN', 'NNP')]

    results["Number of Singular nouns"] = (len(sing_nouns)) if sing_nouns else 0
    results["Ratio of Singular nouns to all nouns"] = len(sing_nouns)/ len(nouns) if sing_nouns else 0


    verbs = [word for word, pos in tags if pos.startswith('VB')]

    results['Number of Verbs'] = len(verbs)
    results['Average verbs per word'] = len(verbs) / len(tokens) if len(tokens) > 0 else 0

    adjectives = [word for word, pos in tags if pos.startswith('JJ')]

    results['Number of Adjectives'] = len(adjectives)
    results['Average adjectives per word'] = len(adjectives) / len(tokens) if len(tokens) > 0 else 0

    adverbs = [word for word, pos in tags if pos.startswith('RB')]

    results['Number of Adverbs'] = len(adverbs)
    results['Average adverbs per word'] = len(adverbs) / len(tokens) if len(tokens) > 0 else 0

    prepositions = [word for word, pos in tags if pos == 'IN']

    results['Number of Prepositions'] = len(prepositions)
    results['Average prepositions per word'] = len(prepositions) / len(tokens) if len(tokens) > 0 else 0

    conjunctions = [word for word, pos in tags if pos.startswith('CC')]

    results['Number of Conjunctions'] = len(conjunctions)
    results['Average conjunctions per word'] = len(conjunctions) / len(tokens) if len(tokens) > 0 else 0

    interjections = [word for word, pos in tags if pos == 'UH']

    results['Number of Interjections'] = len(interjections)
    results['Average interjections per word'] = len(interjections) / len(tokens) if len(tokens) > 0 else 0

    determiners = [word for word, pos in tags if pos == 'DT']

    results['Number of Determiners'] = len(determiners)
    results['Average determiners per word'] = len(determiners) / len(tokens) if len(tokens) > 0 else 0

    pronouns = [word for word, pos in tags if pos.startswith('PRP')]

    results['Number of Pronouns'] = len(pronouns)
    results['Average pronouns per word'] = len(pronouns) / len(tokens) if len(tokens) > 0 else 0

    # syntax
    syn_feats = synt.analyse(text, "eng")
    results["Max Tree depth"] = syn_feats["max_tree_height"] if syn_feats["max_tree_height"] else 0
    results["Max Tree breadth"] = syn_feats["max_tree_breadth"] if syn_feats["max_tree_breadth"] else 0
    results["Average depth"] = syn_feats["avg_depth"] if syn_feats["avg_depth"] else 0
    results["Average breadth"] = syn_feats["avg_breadth"] if syn_feats["avg_breadth"] else 0
    results["Max branching factor"] = syn_feats["max_branching_factor"] if syn_feats["max_branching_factor"] else 0
    results["Average head distance"] = syn_feats["avg_head_distance"] if syn_feats["avg_head_distance"] else 0
    results["Subject ratio"] = syn_feats["subject_ratio"] if syn_feats["subject_ratio"] else 0
    results["Object ratio"] = syn_feats["object_ratio"] if syn_feats["object_ratio"] else 0
    results["Average subject depth"] = np.mean(syn_feats["subject_depths"]) if syn_feats["subject_depths"] else 0
    results["Average object depth"] = np.mean(syn_feats["object_depths"]) if syn_feats["object_depths"] else 0
    results["Average verb depth"] = syn_feats["avg_verb_depth"] if syn_feats["avg_verb_depth"] else 0
    results["Average noun depth"] = syn_feats["avg_noun_depth"] if syn_feats["avg_noun_depth"] else 0
    results["Left-right branching ratio"] = syn_feats["left_right_branching_ratio"] if syn_feats["left_right_branching_ratio"] else 0
    results["Average subject-head distance"] = syn_feats["avg_subj_head_distance"] if syn_feats["avg_subj_head_distance"] else 0
    results["Average object-head distance"] = syn_feats["avg_obj_head_distance"] if syn_feats["avg_obj_head_distance"] else 0
    results["Total clause count"] = syn_feats["clause_count"] if syn_feats["clause_count"] else 0
    results["Clause count per sentence"] = syn_feats["avg_clause_count_per_sent"] if syn_feats["avg_clause_count_per_sent"] else 0
    results["Clause Density (clause per tokens)"] = syn_feats["clause_density"] if syn_feats["clause_density"] else 0
    results["Number of coordinated conjuncts"] = syn_feats["conj_count"] if syn_feats["conj_count"] else 0
    results["Number of auxiliaries"] = syn_feats["aux_count"] if syn_feats["aux_count"] else 0
    results["Average leaf ratio"] = syn_feats["avg_leaf_ratio"] if syn_feats["avg_leaf_ratio"] else 0

    return results


@lru_cache(maxsize=1)
def load_aggregated_stats(path=os.path.join(BASE_DIR, "aggregated_stats.json")):
    """Load aggregated stats from disk once (cached). Returns {} on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning(f"Could not load aggregated stats: {e}")
        return {}

def _normal_cdf(x, mean, std):
    """Normal CDF (returns 0..1). Returns None if std invalid."""
    if std is None or std == 0:
        return None
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def _compute_between_pct(user, a, b):
    """
    Position of user between a (clear mean) and b (not_clear mean) as 0..100.
    Clamps outside range to 0/100. If a==b returns 50.
    """
    if a is None or b is None:
        return None
    if a == b:
        return 50.0
    if a < b:
        ratio = (user - a) / (b - a)
    else:
        # invert if clear mean > not_clear mean (so direction doesn't break)
        ratio = (a - user) / (a - b)
    ratio = max(0.0, min(1.0, ratio))
    return ratio * 100.0

def enrich_results(raw_results, agg_stats):
  clear = agg_stats.get("clear", {})
  not_clear = agg_stats.get("not_clear", {})
  cm = clear.get("mean", {})
  nm = not_clear.get("mean", {})
  cstd = clear.get("std", {})

  enriched = {}
  for k, raw_val in (raw_results.items() if isinstance(raw_results, dict) else []):
        try:
            user_val = float(raw_val)
        except Exception:
            user_val = None

        c_mean = cm.get(k)
        n_mean = nm.get(k)
        c_s = cstd.get(k)

        # position between clear mean and not_clear mean
        pos_pct = None
        if user_val is not None and c_mean is not None and n_mean is not None:
            if c_mean == n_mean:
                pos_pct = 50.0
            else:
                ratio = (user_val - c_mean) / (n_mean - c_mean)
                ratio = max(0.0, min(1.0, ratio))
                pos_pct = ratio * 100.0

        # percentile within clear distribution
        pct_clear = None
        if user_val is not None and c_mean is not None and c_s:
            z = (user_val - c_mean) / (c_s * math.sqrt(2))
            cdf = 0.5 * (1 + math.erf(z))
            pct_clear = cdf * 100.0

        enriched[k] = {
            "user": user_val,
            "clear_mean": c_mean,
            "not_clear_mean": n_mean,
            "pos_pct": pos_pct,
            "pct_clear": pct_clear,
        }

  return enriched
