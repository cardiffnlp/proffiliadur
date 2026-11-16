import subprocess
import os
import csv
import pandas as pd
from collections import Counter
import regex as re
import json
#from lexical_diversity import lex_div as ld

"""
This file:
- computes POS counts and Vocabulary List counts
- relying on CyTag for tokenisation/lemmatisation
"""

# Name of the long-lived CyTag container (started separately, see setup.sh /
# README) that requests are executed against via `docker exec`, rather than
# spinning up a fresh `docker run --rm` container per request.
CYTAG_CONTAINER_NAME = "cytag-service"
CYTAG_TIMEOUT = 30  # seconds


class CyTagServiceError(Exception):
    """Raised when the CyTag container is unreachable, not running, or times out."""
    pass


class POSTagger:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.counters = {
            "tokens" : Counter(),
            "tags" : Counter(),
            "words" : Counter(),
            "numbers": Counter(),
            "noun" : Counter(),
            "verb" : Counter(),
            "adj" : Counter(),
            "adv" : Counter(),
            "prep" : Counter(),
            "prep-uni" : Counter(),
            "prep-in" : Counter(),
            "numeral" : Counter(),
            "article" : Counter(),
            "punctuation" : Counter(),
            "conj" : Counter(),
            "conj-lemmas": Counter(),
            "conj-coord": Counter(),
            "conj-sub": Counter(),
            "pronoun" : Counter(),
            #TYPES of pronouns:
            "rhapers": Counter(), #personal
            "rhadib": Counter(), #dependent
            "rhamedd": Counter(), #possessive
            "rhacys": Counter(), #conjunctive
            "rhagof": Counter(), #interrogative
            "rhadang": Counter(), #demonstrative
            "rhaperth": Counter(), #relative
            "rhaatb": Counter(), #reflexive
            "rhacil": Counter(), #reciprocal
            "rhaamh": Counter(), #indefinite
            "pron-lemma-tag": Counter(),
            "particle" : Counter(),
            "determiner": Counter(),
            #TYPES of determiner:
            "bancynnar": Counter(), #simple-early
            "bandang": Counter(), #demonstrative
            "bangof": Counter(), #interrogative
            "banhwyr": Counter(), #simple-late
            "banmedd": Counter(), #possessive
            "banmeint": Counter(), #quantifier
            "lemma" : Counter(),
            "verb-lemma" : Counter(),
            "verb-coverage" : Counter(),
            "noun-lemma" : Counter(),
            "masc-n" : Counter(), # -n = nouns
            "fem-n" : Counter(),
            "sing-n" : Counter(),
            "pl-n" : Counter(),
            "proper-n": Counter(),
            "ebych": Counter(),
            "pres-v" : Counter(), # -v = verb conjugations
            "gorff-v" : Counter(),
            "amherff-v" : Counter(),
            "dyf-v" : Counter(),
            "gorch-v" : Counter(),
            "amod-v" : Counter(),
            "gorb-v": Counter(),
            "dibdyf-v" : Counter(),
            "amhers-v": Counter(),
            "e-v" : Counter(),
            "perf-v": Counter(),
            "perth-v": Counter(),
            "muts": Counter(),
            "sm":Counter(),
            "am":Counter(),
            "nm":Counter(),
            "hm":Counter(),
            #NEW addons for register analysis
            "second-p":Counter(),
            "third-p":Counter(),
            "plural-adj":Counter(),
            "nomin": Counter(),
            "conjuncts": Counter(),
            "downtoners": Counter(),
            "hedges": Counter(),
            "amplifiers": Counter(),
            "emphatics": Counter(),
            "discourse-parts": Counter(),
            "modals": Counter(),
            "full_instance":[],
            "ins_nouns":Counter(),
            "ins_verbs":Counter(),
            "ins_others":Counter(),
            "ins_adjectives":Counter(),
            "ins_adverbs":Counter(),
            "ins_prepositions":Counter(),
            "ins_particles":Counter(),
            "ins_conjunctions":Counter(),
            "ins_articles":Counter(),
            "ins_pronouns":Counter(),
            "ins_determiners":Counter(),
            "ins_numerals":Counter()
        }

    def reset_counters(self):
        self.counters = {
                    "tokens" : Counter(),
                    "tags" : Counter(),
                    "words" : Counter(),
                    "numbers": Counter(),
                    "noun" : Counter(),
                    "verb" : Counter(),
                    "adj" : Counter(),
                    "adv" : Counter(),
                    "prep" : Counter(),
                    "prep-uni" : Counter(),
                    "prep-in" : Counter(),
                    "numeral" : Counter(),
                    "article" : Counter(),
                    "punctuation" : Counter(),
                    "conj" : Counter(),
                    "conj-lemmas": Counter(),
                    "conj-coord": Counter(),
                    "conj-sub": Counter(),
                    "pronoun" : Counter(),
                    #TYPES of pronouns:
                    "rhapers": Counter(), #personal
                    "rhadib": Counter(), #dependent
                    "rhamedd": Counter(), #possessive
                    "rhacys": Counter(), #conjunctive
                    "rhagof": Counter(), #interrogative
                    "rhadang": Counter(), #demonstrative
                    "rhaperth": Counter(), #relative
                    "rhaatb": Counter(), #reflexive
                    "rhacil": Counter(), #reciprocal
                    "rhaamh": Counter(), #indefinite
                    "pron-lemma-tag": Counter(),
                    "particle" : Counter(),
                    "determiner": Counter(),
                    #TYPES of determiner:
                    "bancynnar": Counter(), #simple-early
                    "bandang": Counter(), #demonstrative
                    "bangof": Counter(), #interrogative
                    "banhwyr": Counter(), #simple-late
                    "banmedd": Counter(), #possessive
                    "banmeint": Counter(), #quantifier
                    "lemma" : Counter(),
                    "verb-lemma" : Counter(),
                    "verb-coverage" : Counter(),
                    "noun-lemma" : Counter(),
                    "masc-n" : Counter(), # -n = nouns
                    "fem-n" : Counter(),
                    "sing-n" : Counter(),
                    "pl-n" : Counter(),
                    "proper-n": Counter(),
                    "ebych": Counter(),
                    "pres-v" : Counter(), # -v = verb conjugations
                    "gorff-v" : Counter(),
                    "amherff-v" : Counter(),
                    "dyf-v" : Counter(),
                    "gorch-v" : Counter(),
                    "amod-v" : Counter(),
                    "gorb-v": Counter(),
                    "dibdyf-v" : Counter(),
                    "amhers-v": Counter(),
                    "e-v" : Counter(),
                    "perf-v": Counter(),
                    "perth-v": Counter(),
                    "muts": Counter(),
                    "sm":Counter(),
                    "am":Counter(),
                    "nm":Counter(),
                    "hm":Counter(),
                    #NEW addons for register analysis
                    "second-p":Counter(),
                    "third-p":Counter(),
                    "plural-adj":Counter(),
                    "nomin": Counter(),
                    "conjuncts": Counter(),
                    "downtoners": Counter(),
                    "hedges": Counter(),
                    "amplifiers": Counter(),
                    "emphatics": Counter(),
                    "discourse-parts": Counter(),
                    "modals": Counter(),
                    "full_instance":[],
                    "ins_nouns":Counter(),
                    "ins_verbs":Counter(),
                    "ins_others":Counter(),
                    "ins_adjectives":Counter(),
                    "ins_adverbs":Counter(),
                    "ins_prepositions":Counter(),
                    "ins_particles":Counter(),
                    "ins_conjunctions":Counter(),
                    "ins_articles":Counter(),
                    "ins_pronouns":Counter(),
                    "ins_determiners":Counter(),
                    "ins_numerals":Counter()
                }

    def tag_count(self, wordform, lemma, simple_tag, pos_tag, mut_tag):

        #General

        self.counters['tokens'][wordform] += wordform.count('_') + 1
        self.counters["tags"][pos_tag] += 1
        self.counters["lemma"][lemma] += 1
        self.counters["full_instance"].append((wordform, simple_tag, lemma))
        if simple_tag != "B" and simple_tag != "E" and simple_tag != "Ans" and simple_tag != "Adf" and simple_tag != "Rha" and simple_tag != "Cys" and simple_tag != "YFB" and simple_tag != "U" and simple_tag != "Ar" and simple_tag != "Ban":
            self.counters["ins_others"][(wordform, lemma, pos_tag, mut_tag)] += 1
        if simple_tag != "Atd":
            self.counters["words"][wordform] += wordform.count('_') + 1
        if wordform.isnumeric():
            self.counters["numbers"][wordform] += 1
        if "2" in pos_tag:
            self.counters['second-p'][pos_tag] += 1
        if "3" in pos_tag:
            self.counters['third-p'][pos_tag] += 1
        modals_ish = ["gallu", "medru", "methu", "rhaid", "angen", "dylai"]
        if lemma in modals_ish:
            self.counters["modals"][lemma] += 1

        #Muts
        mut_tag = mut_tag.strip('+')

        if mut_tag in {"sm", "nm", "am", "hm"}:
            self.counters["muts"][mut_tag] += 1
            if mut_tag == "sm":
                self.counters["sm"][mut_tag] += 1
            elif mut_tag == "nm":
                self.counters["nm"][mut_tag] += 1
            elif mut_tag == "am":
                self.counters["am"][mut_tag] += 1
            elif mut_tag == "hm":
                self.counters["hm"][mut_tag] += 1

        #Verbs

        if simple_tag == "B":
            self.counters["verb"][simple_tag] += 1
            self.counters["verb-coverage"][pos_tag] += 1

            verb_info = pos_tag.strip('B')
            parts = re.split(r'(1|2|3)', verb_info, maxsplit=1)
            if verb_info != 'e' and len(parts) == 3:
                tense, person, number = parts
                self.counters["ins_verbs"][(wordform, lemma, tense, person, number, mut_tag)] += 1
            else:
                self.counters["ins_verbs"][(wordform, lemma, verb_info, '-', '-', mut_tag)] += 1
            if lemma.isalpha():
                self.counters["verb-lemma"][lemma] += 1

            if "pres" in pos_tag:
                self.counters["pres-v"][pos_tag] += 1
            if "gorff" in pos_tag:
                self.counters["gorff-v"][pos_tag] += 1
            if "amherff" in pos_tag:
                self.counters["amherff-v"][pos_tag] += 1
            if "dyf" in pos_tag and "dib" not in pos_tag:
                self.counters["dyf-v"][pos_tag] += 1
            if "gorch" in pos_tag:
                self.counters["gorch-v"][pos_tag] += 1
            if "amod" in pos_tag:
                self.counters["amod-v"][pos_tag] += 1
            if "dibdyf" in pos_tag:
                self.counters["dibdyf-v"][pos_tag] += 1
            if pos_tag == "Be":
                self.counters["e-v"][pos_tag] += 1
            if "gorb" in pos_tag:
                self.counters["gorb-v"][pos_tag] += 1
            if "amhers" in pos_tag:
                self.counters["amhers-v"][pos_tag] += 1
            if "perth" in pos_tag:
                self.counters['perth-v'][pos_tag] += 1

        if lemma == "wedi" and simple_tag == "U":
            self.counters["perf-v"][pos_tag] += 1

        #Nouns

        if simple_tag == "E":
            self.counters["noun"][simple_tag] += 1
            noun_info = pos_tag.strip('E')
            proper = '0'
            if any(x in noun_info for x in ('b', 'g')):
                gender = 'b' if 'b' in noun_info else 'g'
            else:
                gender = '0'
            if 'p' in noun_info:
                proper = 'p'
            if any(x in noun_info for x in ('u', 'll')):
                number = 'u' if 'u' in noun_info else 'll'
            else:
                number = '0'
            self.counters["ins_nouns"][(wordform, lemma, proper, gender, number, mut_tag)] += 1
            nominalisers = ['iad','iant','aeth','der','deb','ineb','rwydd','wch','edd','did','dod']
            if any(wordform.endswith(suffix) for suffix in nominalisers):
                self.counters['nomin'][wordform] += 1

            if lemma.isalpha():
                self.counters["noun-lemma"][lemma] += 1

            if "b" in pos_tag:
                self.counters["fem-n"][pos_tag] += 1
            if "g" in pos_tag:
                self.counters["masc-n"][pos_tag] += 1
            if "u" in pos_tag:
                self.counters["sing-n"][pos_tag] += 1
            if "ll" in pos_tag:
                self.counters["pl-n"][pos_tag] += 1
            if "p" in pos_tag:
                self.counters["proper-n"][pos_tag] += 1

        ebych_discourse_markers = [
            "wel", "reit", "yndê"
        ]

        if pos_tag == "Ebych":
            self.counters["ebych"][pos_tag] += 1
            if lemma in ebych_discourse_markers:
                self.counters["discourse-parts"][wordform] += 1
            

        #ETC

        if simple_tag == "Ans":
            self.counters["adj"][simple_tag] += 1
            adj_type = ''
            if pos_tag == "Anscadu":
                adj_type = "possing"
            elif pos_tag == "Anscadbu":
                adj_type = "posfemsing"
            elif pos_tag == "Anscadll":
                adj_type = "pospl"
                self.counters["plural-adj"]
            elif pos_tag == "Anscyf":
                adj_type = "equative"
            elif pos_tag == "Anscym":
                adj_type = "comparative"
            elif pos_tag == "Anseith":
                adj_type = "superlative"
            self.counters["ins_adjectives"][(wordform, lemma, pos_tag, adj_type, mut_tag)] += 1

        conjuncts = [ #ALL ADF
            "fodd bynnag", "o ganlyniad", "serch hyn", "serch hynny",
            "wedi'r cwbl", "wedi'r cyfan", "yn hytrach","yn hytrach na", "i'r gwrthwyneb",
            "yn ogystal", "at hynny", "ac eithrio", "fel arall", "fel y cyfryw",
            "hynny yw",
            "felly", "hefyd", "'efyd", "ychwaith", "eto"
        ]

        downtoners = [
            "go brin", "o leiaf", "i raddau", "i raddau helaeth", "go lew",
            "braidd", "lled", "rhywfaint"
        ]

        hedges = [
            "o bosib", "yn fras", "mwy na thebyg", "eitha debyg", "fel petai",
            "fath a", "siŵr o fod", "duw a ŵyr", "pwy a ŵyr",
            "bron", "tua", #"ryw"
        ]

        amplifiers = [
            "yn gyfan gwbl", "i'r dim", "dros ben", "heb os", "heb os nac onibai",
            "o ddifrif", "llygad ei le",
            "hollol", "llwyr", "eithriadol", "aruthrol", "hynod", "iawn", "tra"
        ]

        emphatics = [
            "gwir yr", "wrth gwrs", "wrth reswm", "mewn gwirionedd", "go iawn",
            "wir", "siŵr"
        ]

        adf_discourse_particles = [
            "nawr te", "beth bynnag", "ta waeth", "gyda llaw", "erbyn meddwl", "dyma", "dyna"
        ]

        if simple_tag == "Adf":
            self.counters["adv"][simple_tag] += 1
            self.counters["ins_adverbs"][(wordform, lemma, pos_tag, mut_tag)] += 1
            if lemma in conjuncts:
                self.counters["conjuncts"][wordform] += 1
            if lemma in downtoners:
                self.counters["downtoners"][wordform] += 1
            if lemma in hedges:
                self.counters["hedges"][wordform] += 1
            if lemma in amplifiers:
                self.counters["amplifiers"][wordform] += 1
            if lemma in emphatics:
                self.counters["emphatics"][wordform] += 1
            if lemma in adf_discourse_particles:
                self.counters["discourse-parts"][wordform] += 1

        if simple_tag == "Ar":
            self.counters["prep"][simple_tag] += 1
            prep_type = ""
            if 'sym' in pos_tag:
                self.counters["prep-uni"][pos_tag] += 1
                prep_type = "uninflected"
            else:
                self.counters["prep-in"][pos_tag] += 1
                prep_type = "inflected"
            self.counters["ins_prepositions"][(wordform, lemma, pos_tag, prep_type)] += 1

        if simple_tag == "Rhi":
            self.counters["numeral"][simple_tag] += 1
            self.counters["ins_numerals"][(wordform, lemma, pos_tag)] += 1

        if simple_tag == "YFB":
            self.counters["article"][simple_tag] += 1
            self.counters["ins_articles"][(wordform, lemma, pos_tag)] += 1

        if simple_tag == "Atd":
            self.counters["punctuation"][simple_tag] += 1

        if simple_tag == "Cys":
            self.counters["conj"][simple_tag] += 1
            conj_type = ''

            if lemma.isalpha():
                self.counters["conj-lemmas"][lemma] += 1
            if pos_tag == "Cyscyd":
                self.counters["conj-coord"][lemma] += 1
                conj_type = "coordinating"
            if pos_tag == "Cysis":
                self.counters["conj-sub"][lemma] += 1
                conj_type = "subordinating"
            self.counters["ins_conjunctions"][(wordform, lemma, pos_tag, conj_type)] += 1

        if simple_tag == "Rha":
            self.counters["pronoun"][simple_tag] += 1
            pron_type = ""
            if "pers" in pos_tag:
                pron_type = "personal"
                self.counters["rhapers"][wordform] += 1
            elif "dib" in pos_tag:
                pron_type = "dependent"
                self.counters["rhadib"][wordform] += 1
            elif "medd" in pos_tag:
                pron_type = "possessive"
                self.counters["rhamedd"][wordform] += 1
            elif "cys" in pos_tag:
                pron_type = "conjunctive"
                self.counters["rhacys"][wordform] += 1
            elif "gof" in pos_tag:
                pron_type = "interrogative"
                self.counters["rhagof"][wordform] += 1
            elif "dang" in pos_tag:
                pron_type = "demonstrative"
                self.counters["rhadang"][wordform] += 1
            elif "perth" in pos_tag:
                pron_type = "relative"
                self.counters["rhaperth"][wordform] += 1
            elif "atb" in pos_tag:
                pron_type = "reflexive"
                self.counters["rhaatb"][wordform] += 1
            elif "cil" in pos_tag:
                pron_type = "reciprocal"
                self.counters["rhacil"][wordform] += 1
            elif "amh" in pos_tag:
                pron_type = "impersonal"
                self.counters["rhaamh"][wordform] += 1
            self.counters["ins_pronouns"][(wordform, lemma, pos_tag, pron_type)] += 1
            self.counters["pron-lemma-tag"][(lemma, pos_tag)] += 1

        if simple_tag == "U":
            u_type = ""
            self.counters["particle"][simple_tag] += 1
            if "neg" in pos_tag:
                u_type = "negative"
            elif "cad" in pos_tag:
                u_type = "affirmative"
            elif "gof" in pos_tag:
                u_type = "interrogative"
            elif "tra" in pos_tag:
                u_type = "predicative"
            elif "berf" in pos_tag:
                u_type = "verbal"
            self.counters["ins_particles"][(wordform, lemma, pos_tag, u_type)] += 1
        if simple_tag == "Ban":
            self.counters["determiner"][simple_tag] += 1
            det_type = ""
            if "cynnar" in pos_tag:
                self.counters["bancynnar"][wordform] += 1
                det_type = "Simple-early"
            if "dang" in pos_tag:
                self.counters["bandang"][wordform] += 1
                det_type = "demonstrative"
            if "gof" in pos_tag:
                self.counters["bangof"][wordform] += 1
                det_type = "interrogative"
            if "hwyr" in pos_tag:
                self.counters["banhwyr"][wordform] += 1
                det_type = "simple-late"
            if "medd" in pos_tag:
                self.counters["banmedd"][wordform] += 1
                det_type = "possessive"
            if "meint" in pos_tag:
                self.counters["banmeint"][wordform] += 1
                det_type = "quantifier"
            self.counters["ins_determiners"][(wordform, lemma, pos_tag, det_type)] += 1

    def parser(self, text):
        """Run CyTag via `docker exec` against the persistent cytag-service
        container (started separately - see setup.sh), using stdin and
        parsing stdout. Avoids the ~30-40% latency cost of spinning up a
        fresh `docker run --rm` container on every request."""
        text_bytes = text.encode("utf-8", errors="ignore")

        try:
            result = subprocess.run(
                [
                    "docker", "exec", "-i",
                    CYTAG_CONTAINER_NAME,
                    "/usr/nobody/venv/bin/python", "CyTag.py"
                ],
                input=text_bytes,
                capture_output=True,
                text=False,
                timeout=CYTAG_TIMEOUT,
                check=True
            )
        except subprocess.TimeoutExpired as e:
            raise CyTagServiceError(f"CyTag timed out after {CYTAG_TIMEOUT}s") from e
        except FileNotFoundError as e:
            raise CyTagServiceError("Docker is not installed or not on PATH") from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            raise CyTagServiceError(
                f"CyTag container '{CYTAG_CONTAINER_NAME}' failed (is it running? see setup.sh): {stderr}"
            ) from e

        stdout_text = result.stdout.decode("utf-8", errors="replace")
        rows = []

        for line in stdout_text.splitlines():
            line = line.strip()
            #print(line)
            sections = line.split("\t")

            if len(sections) > 5:
                rows.append({
                    "wordform": sections[1],
                    "lemma": sections[3],
                    "simple_tag": sections[4],
                    "pos_tag": sections[5],
                    "mut_tag": sections[6] if len(sections) > 6 else "0m"
                })

        return pd.DataFrame(rows)

    def tagger(self, text):
        self.reset_counters()
        # Uses the parser and calls tag count
        df = self.parser(text)
        if df.empty:
            return self.counters

        for _, row in df.iterrows():

            self.tag_count(wordform=row["wordform"],
                        lemma=row["lemma"],
                        simple_tag=row["simple_tag"],
                        pos_tag=row["pos_tag"],
                        mut_tag=row["mut_tag"])

        return self.counters

    # HELPERS

    def csv_to_arr(self, vocab_list):
        with open(vocab_list, "r", encoding="utf-8") as CSV:
            vocab_words = []
            for row in csv.reader(CSV):
                if row:
                    vocab_words.append(row[0].strip().lower())
        return vocab_words

    def stopwords_to_arr(self, stopwords_list):
        with open(stopwords_list, 'r', encoding='utf-8') as stp:
            return [line.strip().lower() for line in stp if line.strip()]

    def frequency_band(self, rank):
        if rank <= 1000:
            return "K1"
        elif rank <= 2000:
            return "K2"
        elif rank <= 3000:
            return "K3"
        elif rank <= 4000:
            return "K4"
        elif rank <= 5000:
            return "K5"
        # else:
        #     return "K6+"

    def json_to_vocab_set(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            entries = json.load(f)
        vocab = set()
        for entry in entries:
            vocab.add(entry["word"].lower())
            vocab.update(v.lower() for v in entry.get("variants", []))
            # collocations ignored for now
        return vocab

    #--------------------

    #Vocab analysis - using wordforms or lemmas tokenised by Cytag

    def analyse_vocab(self, text, vocab_files, band_csv=os.path.join("data", "corcencc_bands.csv"), stopwords_files=None):
        band_csv = os.path.join(self.BASE_DIR, band_csv)
        self.reset_counters()
        df = self.parser(text)
        #print(df)
        if df.empty:
                return self.counters
        
        for _, row in df.iterrows():
        
            self.tag_count(wordform=row["wordform"],
                                lemma=row["lemma"],
                                simple_tag=row["simple_tag"],
                                pos_tag=row["pos_tag"],
                                mut_tag=row["mut_tag"])
        #Lexical diversity

        tokenised_list = df["wordform"].tolist()
        #hdd = ld.hdd(tokenised_list) remove for now til pkg_resources issue can be resolved??
        #mtld = ld.mtld(tokenised_list)


        #CEFR level:

        df["level"] = ""
        lemma_lower = df["lemma"].str.lower()
        for level, file_path in vocab_files.items():
            vocab_words = self.json_to_vocab_set(file_path)
            mask = lemma_lower.isin(vocab_words) & (df["level"] == "")
            df.loc[mask, "level"] = level
        print(df)
        #CorCenCC band:

        df["band"] = None
        corcencc_band = pd.read_csv(band_csv, usecols=[0,1])
        corcencc_band.columns = ["Rank", "Word"]
        corcencc_band["Word"] = corcencc_band["Word"].str.lower()
        corcencc_band["Band"] = corcencc_band["Rank"].apply(self.frequency_band)
        band_lookup = dict(zip(corcencc_band["Word"], corcencc_band["Band"]))

        for i, row in df.iterrows():
            lemma = str(row["wordform"]).lower() #do include wordforms
            if lemma in band_lookup:
                df.at[i, "band"] = band_lookup[lemma]

        #Stopwords:

        results_stopwords = {}

        if stopwords_files:
            for stop_type, file_path in stopwords_files.items():
                col_name = f"stopword_{stop_type}"
                df[col_name] = False  # boolean column per list

                stop_set = set(self.stopwords_to_arr(file_path))
                for i, row in df.iterrows():
                    wordform = str(row["wordform"]).lower()
                    lemma = str(row["lemma"]).lower()
                    if wordform in stop_set or lemma in stop_set:
                        df.at[i, col_name] = True

                # collect per-stoplist stats
                matches = df[df[col_name]]
                unique_lemmas = matches["lemma"].dropna().str.lower().unique()
                results_stopwords[stop_type] = {
                    "unique_stopwords": len(unique_lemmas),
                    "total_appearances": len(matches),
                }

        #Build CEFR results:

        results_cefr = {}
        for level in vocab_files:
            matches = df[df["level"] == level]
            word_counts = matches["lemma"].value_counts().to_dict()


            results_cefr[level] = {
                "unique_matches": len(word_counts),
                "total_appearances": sum(word_counts.values())
            }

        # --- Frequency band results ---
        results_band = {}
        all_bands = ["K1", "K2", "K3", "K4", "K5"]
        for band in all_bands:
            matches = df[df["band"] == band]
            word_counts = matches["lemma"].value_counts().to_dict()
            results_band[band] = {
                "unique_matches": len(word_counts),
                "total_appearances": sum(word_counts.values())
            }
        return df, results_cefr, results_band, results_stopwords, self.counters