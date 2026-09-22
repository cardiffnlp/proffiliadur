import logging
import requests
from conllu import parse
from collections import defaultdict
import numpy as np

"""
This file:
- computes syntactic features
- by posting a request to the UDPipe online server and parsing the response
"""

logger = logging.getLogger(__name__)

# (connect timeout, read timeout) in seconds for the UDPipe request.
UDPIPE_TIMEOUT = (5, 30)


class UDPipeServiceError(Exception):
    """Raised when the external UDPipe service fails, times out, or is unreachable."""
    pass


class SyntaxAnalyser:
    def __init__(self):
        self.cy_model = "welsh-ccg-ud-2.17-251125"
        self.eng_model = "english-atis-ud-2.17-251125"
        self.url = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
    def get_parsed(self, text, lang):
        if lang == "cy":
            data = {
                'tokenizer': 'true',
                'tagger': 'true',
                'parser': 'true',
                'model': self.cy_model,
                'data': text}
        else:
            data = {
                'tokenizer': 'true',
                'tagger': 'true',
                'parser': 'true',
                'model': self.eng_model,
                'data': text}

        try:
            response = requests.post(self.url, data=data, timeout=UDPIPE_TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise UDPipeServiceError(f"UDPipe service unreachable or timed out: {e}") from e

        if response.status_code == 200:
            result = response.json()

            # Check keys
            conllu_text = result.get('result', '')
            if not conllu_text.strip():
                logger.warning("UDPipe returned an empty CONLLU response")

            return conllu_text
        else:
            logger.error(f"UDPipe request failed - response text: {response.text}")
            raise UDPipeServiceError(f"UDPipe request failed with status code {response.status_code}")
    
    def build_tree_dict(self, sentence):
        tree = defaultdict(list)
        all_ids = set()
        for token in sentence:
            if isinstance(token['id'], int):
                all_ids.add(token['id'])
            if isinstance(token['head'], int) and token['head'] != 0:
                tree[token['head']].append(token['id'])

        # Ensure all nodes exist in the tree (even if they have no children)
        for tid in all_ids:
            tree[tid]  # triggers creation of empty list if not present

        return tree
    
    def compute_depths(self, tree):
        depths = {}

        # Infer root node: a node that is never a child
        all_nodes = set(tree.keys())
        all_children = {child for children in tree.values() for child in children}
        root_candidates = all_nodes - all_children

        if not root_candidates:
            logger.warning(f"No root candidate found - tree keys: {tree.keys()}, tree values: {tree}")
            return depths  # fallback if no root found

        root = root_candidates.pop()

        def dfs(node, depth):
            depths[node] = depth
            for child in tree.get(node, []):
                dfs(child, depth + 1)

        dfs(root, 0)
        return depths

    def compute_tree_features(self, conllu):
        sentences = parse(conllu)
        if not sentences:
            logger.warning(f"No sentences parsed from CONLLU. Raw CONLLU (truncated): {conllu[:500]}")
            return {}

        total_depth_values = []
        total_subj_count = 0
        total_obj_count = 0
        total_tokens = 0
        total_head_distance = 0
        total_branching_factors = []
        breadths = []

        all_subj_depths = []
        all_obj_depths = []
        all_verb_depths = []

        total_left_deps = 0
        total_right_deps = 0

        total_clause_count = 0
        total_conj_count = 0
        total_aux_count = 0
        total_leaf_ratios = []

        subj_distances = []
        obj_distances = []

        pos_depths = defaultdict(list)

        for sentence in sentences:
            tree = self.build_tree_dict(sentence)
            if not tree:
                logger.warning("Empty tree structure")
            depths = self.compute_depths(tree)
            if not depths:
                logger.warning(f"No depths computed (likely no root) - tree: {dict(tree)}")
            depth_values = list(depths.values())
            breadth = max(len(children) for children in tree.values())

            total_depth_values.extend(depth_values)
            breadths.append(breadth)
            total_branching_factors.append(max(len(children) for children in tree.values()))

            deprels = [token['deprel'] for token in sentence]
            subj_count = sum(1 for rel in deprels if 'subj' in rel)
            obj_count = sum(1 for rel in deprels if 'obj' in rel)
            total_subj_count += subj_count
            total_obj_count += obj_count

            subj_depths = [
                depths[token['id']]
                for token in sentence
                if 'subj' in token['deprel'] and isinstance(token['id'], int) and token['id'] in depths
            ]
            obj_depths = [
                depths[token['id']]
                for token in sentence
                if 'obj' in token['deprel'] and isinstance(token['id'], int) and token['id'] in depths
            ]
            verb_depths = [
                depths[token['id']]
                for token in sentence
                if token['upos'] == 'VERB' and isinstance(token['id'], int) and token['id'] in depths
            ]
            all_subj_depths.extend(subj_depths)
            all_obj_depths.extend(obj_depths)
            all_verb_depths.extend(verb_depths)

            for token in sentence:
                if isinstance(token['id'], int) and token['id'] in depths:
                    pos_depths[token['upos']].append(depths[token['id']])

            left_deps = sum(1 for token in sentence if isinstance(token['head'], int) and token['id'] < token['head'])
            right_deps = sum(1 for token in sentence if isinstance(token['head'], int) and token['id'] > token['head'])
            total_left_deps += left_deps
            total_right_deps += right_deps

            clause_count = sum(1 for token in sentence if token['deprel'] in {'advcl', 'ccomp', 'xcomp'})
            total_clause_count += clause_count

            conj_count = sum(1 for token in sentence if token['deprel'] == 'conj')
            total_conj_count += conj_count

            aux_count = sum(1 for token in sentence if token['deprel'] == 'aux')
            total_aux_count += aux_count

            num_leaves = sum(1 for node, children in tree.items() if not children)

            leaf_ratio = num_leaves / len(tree) if tree else 0
            total_leaf_ratios.append(leaf_ratio)

            total_tokens += len(sentence)
            for token in sentence:
                if isinstance(token['head'], int) and token['head'] != 0:
                    total_head_distance += abs(token['id'] - token['head'])
                    if 'subj' in token['deprel']:
                        subj_distances.append(abs(token['id'] - token['head']))
                    if 'obj' in token['deprel']:
                        obj_distances.append(abs(token['id'] - token['head']))

        features = {
            "max_tree_height": max(total_depth_values, default=0),
            "max_tree_breadth": max(breadths, default=0),
            "avg_depth": sum(total_depth_values) / len(total_depth_values) if total_depth_values else 0,
            "avg_breadth": sum(breadths)/len(breadths) if breadths else 0,
            "max_branching_factor": max(total_branching_factors, default=0),

            "avg_head_distance": total_head_distance / total_tokens if total_tokens else 0,

            "subject_ratio": total_subj_count / total_tokens if total_tokens else 0,
            "object_ratio": total_obj_count / total_tokens if total_tokens else 0,
            "subject_depths": all_subj_depths,
            "object_depths": all_obj_depths,
            "verb_depths": all_verb_depths,

            "left_right_branching_ratio": total_left_deps / (total_right_deps + 1e-5),
            "avg_subj_head_distance": np.mean(subj_distances) if subj_distances else 0,
            "avg_obj_head_distance": np.mean(obj_distances) if obj_distances else 0,
            "clause_count": total_clause_count if total_clause_count else 0,
            "avg_clause_count_per_sent": total_clause_count / len(sentences) if sentences else 0,
            "clause_density": total_clause_count / total_tokens if total_tokens else 0,


            "conj_count": total_conj_count,
            "aux_count": total_aux_count,
            "avg_leaf_ratio": np.mean(total_leaf_ratios) if total_leaf_ratios else 0,
        }

        for pos in ['VERB', 'NOUN']:
            key = f'avg_{pos.lower()}_depth'
            features[key] = np.mean(pos_depths[pos]) if pos_depths[pos] else 0
        return features       

            
    def analyse(self, text, lang):
        conllu = self.get_parsed(text, lang)
        features = self.compute_tree_features(conllu)
        #print(features)
        return features