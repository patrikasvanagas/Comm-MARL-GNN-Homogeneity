"""
Compute the similarity of the embeddings of different agents.
As of 11 March 2024, we assume the embeddings are of the shape:
    nr_agents x nr_samples x nr_parallel_envs x dimension_embeddings
"""
import math

import numpy as np


def cosine_similarity(embeddings_1, embeddings_2):
    """
    Compute the cosine similarity between two sets of embeddings.
    The formula is:
        similarity(A, B) = A dot B / ( ||A|| * ||B||)
        where `dot` is the dot product, `||X||` is the magnitude of X.
    """

    dot_product = np.dot(embeddings_1, embeddings_2)

    magnitude_1 = np.linalg.norm(embeddings_1)
    magnitude_2 = np.linalg.norm(embeddings_2)

    return dot_product / (magnitude_1 * magnitude_2)


def similarity(embeddings):
    """
    :param embeddings: The embeddings of the agents of shape [nr_agents, nr_samples, nr_parallel_envs, dim_embeddings]
    """
    nr_agents, nr_samples, nr_parallel_envs, dim_embeddings = embeddings.shape
    similarities = np.zeros((nr_samples, nr_parallel_envs))
    for sample in range(nr_samples):
        for env in range(nr_parallel_envs):
            sample_env_similarities = []
            for i in range(nr_agents):
                for j in range(nr_agents):
                    if j < i:
                        s = cosine_similarity(embeddings[i, sample, env], embeddings[j, sample, env])
                        sample_env_similarities.append(s)
            # We want the pairwise (k=2 in combination formula) similarity for all pairs where i != j, w/o repetitions
            nr_possible_pairs = int(math.factorial(nr_agents) / (2 * math.factorial(nr_agents-2)))
            assert len(sample_env_similarities) == nr_possible_pairs

            similarities[sample, env] = np.mean(sample_env_similarities)

    assert similarities.shape == (nr_samples, nr_parallel_envs)
    return similarities




