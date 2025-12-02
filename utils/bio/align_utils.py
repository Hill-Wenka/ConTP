import numpy as np
import pandas as pd
from copy import copy
from itertools import count
from Bio.Align import PairwiseAligner, substitution_matrices
from utils.parallel import concurrent_submit
from utils.tool import merge_dicts

# Substitution matrix for computing similarity score
BLOSUM62 = substitution_matrices.load("BLOSUM62")
PAM250 = substitution_matrices.load("PAM250")


def identity(seq1, seq2, align=False):
    if align:
        # 使用全局比对计算序列相似度
        return align_pair(seq1, seq2).identity
    else:
        assert len(seq1) == len(seq2), 'The length of the two sequences must be equal.'
        return sum([bool(a == b) for a, b in zip(seq1, seq2)]) / len(seq1)


def align_pair(seq1, seq2, scoring='blastp', matrix=None, score_only=False, **kwargs):
    if scoring is not None:
        assert scoring in ["blastn", "megablast", "blastp"]

    if scoring is None:
        if matrix is None:
            kwargs.setdefault('match_score', 1)
            kwargs.setdefault('mismatch_score', 0)
        else:
            if matrix == 'blosum62':
                matrix = BLOSUM62
            elif matrix == 'pam250':
                matrix = PAM250
            else:
                assert not isinstance(matrix, str)
            kwargs.setdefault('open_gap_score', -10)
            kwargs.setdefault('extend_gap_score', -0.5)
    else:
        assert matrix is None, "If scoring is specified, matrix will be ignored."

    aligner = PairwiseAligner(scoring, **kwargs)
    aligner.substitution_matrix = matrix

    if score_only:
        score = aligner.score(seq1, seq2)  # float
        return score
    else:
        alignment = aligner.align(seq1, seq2)[0]  # return the first alignment by default
        seq1_aligned, seq2_aligned = alignment.__array__()
        align_identity = sum(seq1_aligned == seq2_aligned) / len(seq1_aligned)
        num_diff = sum(seq1_aligned != seq2_aligned)
        alignment.__setattr__('identity', align_identity)
        alignment.__setattr__('num_diff', num_diff)
        return alignment


def align_attribute(aligned_seq, unaligned_attribute):
    aligned_attribute = []
    j = 0  # index for original_attr
    for aa in aligned_seq:
        if aa == '-':
            aligned_attribute.append('-')
        else:
            aligned_attribute.append(unaligned_attribute[j])
            j += 1
    return aligned_attribute


def batch_identity(sequence, seq_list, align=True):
    identity_list = []
    for j, seq_j in enumerate(seq_list):
        seq_j_identity = identity(sequence, seq_j, align=align)
        identity_list.append(seq_j_identity)
    return {sequence: identity_list}


def format_alignment(alignment, seq_names=None):
    aa_counter = count(1)
    ref_seq = alignment[0]
    column = [next(aa_counter) if aa != '-' else '-' for aa in ref_seq]
    align_df = pd.DataFrame(alignment.__array__(), index=seq_names, columns=column, dtype='str')
    return align_df


def novelty(sequences, references, transform=True, return_list=False, rounding=3):
    params = [(seq, references) for seq in sequences]
    results = concurrent_submit(batch_identity, params)
    results = merge_dicts(results)

    similarity_list = []
    for seq in sequences:
        max_identity = max(results[seq])
        value = (1 - max_identity) if transform else max_identity
        similarity_list.append(value)
    if return_list:
        return similarity_list
    else:
        return round(np.mean(similarity_list), rounding)


def diversity(sequences, transform=True, return_list=False, rounding=3):
    references = copy(sequences)
    params = [(seq_i, references) for seq_i in sequences]
    results = concurrent_submit(batch_identity, params)
    results = merge_dicts(results)

    similarity_list = []
    for seq in sequences:
        for x in results[seq]:
            value = (1 - x) if transform else x
            similarity_list.append(value)
    if return_list:
        return similarity_list
    else:
        return round(np.mean(similarity_list), rounding)
