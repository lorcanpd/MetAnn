import tensorflow as tf
import random
from collections import defaultdict, deque
import numpy as np
import networkx as nx


class Adjacency():

    def __init__(self, relation_dicts, concept2index, config):
        super(Adjacency, self).__init__()
        self.config = config
        self.concept_dim = config.concept_dim
        self.max_hops = config.max_path_len
        self.num_nodes = len(concept2index)
        self.sample_size = config.max_neighbourhood_sample
        self.negative_sample_rate = config.negative_sample_rate
        self.concept2index = concept2index
        self.concept2index['[CLS]'] = len(concept2index)
        self.index2concept = {v: k for k, v in self.concept2index.items()}
        # replace strings with index integers
        self.relation_dicts = self.transform_relation_dicts(relation_dicts)

        reverse_dicts = []
        for relation_dict in self.relation_dicts:
            reverse_dict = self.create_reverse_relation_dict(relation_dict)
            reverse_dicts.append(reverse_dict)

        self.relation_dicts.extend(reverse_dicts)
        self.combine_relation_dicts()
        self.get_degree_centralities()
        # self.plot_nodes_within_n_hops()
        # self.plot_nodes_within_n_hops_of_hpo()


    def transform_relation_dicts(self, relation_dicts):
        """Transforms the relation dicts to use the index of the concepts. This
        should make the adjacency matrix smaller and faster to compute."""
        transformed_dicts = []
        for rel_dict in relation_dicts:
            transformed_dict = {}
            for entity, related_entities in rel_dict.items():
                transformed_entity = self.concept2index.get(entity)
                transformed_related_entities = [
                    self.concept2index.get(e) for e in related_entities
                ]
                transformed_dict[
                    transformed_entity] = transformed_related_entities
            transformed_dicts.append(transformed_dict)
        return transformed_dicts

    @staticmethod
    def create_reverse_relation_dict(relation_dict):
        reverse_dict = defaultdict(list)
        for key, values in relation_dict.items():
            for value in values:
                reverse_dict[value].append(key)
        return dict(reverse_dict)

    def get_degree_centralities(self):
        """Computes the degree centralities of the nodes in the adjacency matrix.
        """
        graph = nx.DiGraph(self.all_relations)
        centralities = nx.degree_centrality(graph)
        sampling_probs = {node: 1.0 / score for node, score in 
                          centralities.items()}
        normalising_factor = sum(sampling_probs.values())
        self.sampling_probs = {
            node: score / normalising_factor
            for node, score in sampling_probs.items()
        }
        self.degree_centralities = centralities
        self.degrees = dict(graph.degree())

    def combine_relation_dicts(self):
        self.all_relations = defaultdict(set)
        for relation_dict in self.relation_dicts:
            for key, value in relation_dict.items():
                self.all_relations[key].update(value)
        self.all_relations = {
            k: v for k, v in self.all_relations.items() if k is not None
        }

    def batch_nodes_within_n_hops(self, start_nodes, hops, training=False):
        batch = []
        if isinstance(start_nodes, tf.Tensor):
            start_nodes = start_nodes.numpy()
        if training:
            sample_size = self.sample_size/4
        else:
            sample_size = self.sample_size

        for i, node in enumerate(start_nodes):
            within_hops = list(self.get_nodes_within_n_hops(
                node, hops, sample_size
            ))
            
            if training:
                # include other hpo nodes in each sample so that the model can
                # learn to distinguish between them.
                for _ in range(4):
                    negative = random.choice(
                        [x for idx, x in enumerate(start_nodes) if idx != i]
                    )
                    within_hops.extend(list(self.get_nodes_within_n_hops(
                        negative, hops, sample_size
                    )))

            within_hops = [node] + [x for x in within_hops if x != node]
            batch.append(within_hops[:self.sample_size])

        return batch

    def random_walk(self, start_node, max_hops, visited):
        path = [start_node]
        for _ in range(max_hops):
            if path[-1] in self.all_relations:
                unvisited_children = list(
                    set(self.all_relations[path[-1]]) - set(visited)
                )
                if unvisited_children:
                    next_node = random.choices(
                        unvisited_children,
                        weights=[self.sampling_probs[node] for node
                                 in unvisited_children]
                    )[0]
                    path.append(next_node)
                else:
                    break
            else:
                break
        return path

    def recursive_walk(self, start_node, hops, sample_size,
                       distance_from_original_start=0):
        if len(self.sampled) >= sample_size:
            return

        self.visited[start_node] = distance_from_original_start
        remaining_hops = hops - distance_from_original_start
        random_walk_nodes = self.random_walk(start_node, remaining_hops,
                                             self.visited)
        for i, node in enumerate(random_walk_nodes):
            if node not in self.visited:
                self.visited[node] = i + distance_from_original_start
                if node != start_node:
                    self.sampled.add(node)
                if len(self.sampled) >= sample_size:
                    break

        unvisited_children = set(
            self.all_relations.get(start_node, [])
        ) - self.visited.keys()
        if unvisited_children:
            children = list(unvisited_children)
            next_start_node = random.choices(
                children,
                weights=[self.sampling_probs[node] for node in children]
            )[0]
            self.recursive_walk(
                next_start_node, hops, sample_size,
                distance_from_original_start + 1
            )

    def get_nodes_within_n_hops(self, start_node, hops, sample_size):
        self.visited = {start_node: 0}
        self.sampled = {start_node}

        self.recursive_walk(start_node, hops, sample_size, 0)

        return self.sampled

    @staticmethod
    def get_adjacency_matrix(relation_dict, node_list):
        """Returns an adjacency matrix for the given entities and relation 
        dictionary."""
        num_nodes = len(node_list)
        adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        # row child, col parent
        for i, parent in enumerate(node_list):
            for j, child in enumerate(node_list):
                if parent == child:
                    adjacency_matrix[j][i] = 0
                elif parent in relation_dict and child in \
                        relation_dict[parent]:
                    adjacency_matrix[j][i] = 1

        return tf.constant(adjacency_matrix, dtype=tf.float32)


    def compile_adjacency_matrices(self, node_list):
        """Compiles the local adjacency matrices using each relation
        dictionary."""
        if isinstance(node_list, tf.Tensor):
            node_list = node_list.numpy()
        adjacency_matrices = []
        for relation_dict in self.relation_dicts:
            adjacency_matrices.append(
                self.get_adjacency_matrix(relation_dict, node_list)
            )
        identity = [tf.eye(len(node_list), dtype=tf.float32)]

        return tf.stack(identity + adjacency_matrices, axis=0)

    def batch_adjacency_matrices(self, list_of_node_lists):
        """Compiles the local adjacency matrices for a batch of nodes."""
        adjacency_matrices = []
        degree_matrices = []
        if not isinstance(list_of_node_lists[0], list):
            list_of_node_lists = [list_of_node_lists]
        for node_list in list_of_node_lists:
            # compile adjacency matrices for the given node list
            adj_matrices = self.compile_adjacency_matrices(node_list)
            # if the adjacency matrix is too small for the desired sample size,
            # pad it with zeros
            if len(node_list) < self.sample_size:
                pad_width = [(0, 0)] * (adj_matrices.ndim - 2) + [
                    (0, self.sample_size - len(node_list))] * 2

                padded_adj_matrices = np.pad(adj_matrices, pad_width,
                                             mode='constant')
            else:
                padded_adj_matrices = adj_matrices
            # add padded adjacency matrices to the list
            adjacency_matrices.append(padded_adj_matrices)

            # pad node list with 0s to sample size if necessary
            if len(node_list) < self.sample_size:
                node_list += [0] * (self.sample_size - len(node_list))
            # degree_matrix = self.get_degree_matrix(node_list)
            degree_matrix = tf.linalg.diag(
                tf.reduce_sum(
                    tf.reduce_sum(padded_adj_matrices, axis=-3),
                    axis=-1
                ) + 1e-12
            )
            degree_matrices.append(degree_matrix)
        # stack adjacency matrices and return
        return tf.stack(adjacency_matrices, axis=0), tf.stack(degree_matrices, axis=0)

    def pad_node_list_batch(self, batch):
        """Pads the node list batch to the desired size."""
        max_length = max(len(node_list) for node_list in batch)
        # check batch is list of lists to deal with single node lists
        if not isinstance(batch[0], list):
            batch = [batch]
        padded_batch = []
        for node_list in batch:
            padding = [[0, self.sample_size - len(node_list)]]
            padded_node_list = tf.pad(node_list, padding)
            padded_batch.append(padded_node_list)
        return tf.stack(padded_batch)

    def get_random_walk(self, start_node):
        """
        Returns a random walk from a random start entity, following
        relationships in the combined_dict, with a maximum length of max_length.
        """
        walk = [self.concept2index['[CLS]'], start_node]
        current_node = start_node
        for i in range(self.sample_size - 2):
            if current_node in self.all_relations:
                related_nodes = self.all_relations[current_node]
                # Use sampling probabilities
                next_node = random.choices(
                    list(related_nodes),
                    weights=[self.sampling_probs[node]
                             for node in related_nodes]
                )[0]
                walk.append(next_node)
                current_node = next_node
            else:
                break
        return walk

    def get_batch_random_walks(self, batch_size):
        """
        Returns a batch of random walks, where each walk is either a real walk
        or a random walk.

        Args:
        - batch_size: int, the number of walks to generate

        Returns:
        - batch_walks: tf.Tensor of shape (batch_size, self.sample_size), the
        batch of random walks
        """
        labels = []
        batch_walks = []
        adjacency = []
        degree = []
        sampling_keys = list(self.sampling_probs.keys())
        sampling_values = list(self.sampling_probs.values())
        sampled_nodes = random.choices(sampling_keys,
                                       weights=sampling_values,
                                       k=batch_size)

        for node in sampled_nodes:
            walk = self.get_random_walk(node)
            if len(walk) < self.sample_size:
                walk += [0] * (self.sample_size - len(walk))

            # degree.append(self.get_degree_matrix(walk))
            if random.random() > self.negative_sample_rate:
                adjacency.append(self.compile_adjacency_matrices(walk))
                labels.append(1)
            else:
                walk = [
                    x if random.random() > 0.1
                    else random.choices(
                        sampling_keys, weights=sampling_values, k=1
                    )[0]
                    for x in walk
                ]
                for_incorrect_adjacency = self.get_random_walk(
                    random.choices(
                        sampling_keys, weights=sampling_values, k=1
                    )[0]
                )
                adjacency.append(
                    self.compile_adjacency_matrices(for_incorrect_adjacency)
                )
                labels.append(0)
            batch_walks.append(walk[:self.sample_size])

            degree.append(
                tf.linalg.diag(
                    tf.reduce_sum(
                        tf.reduce_sum(adjacency[-1], axis=-3),
                        axis=-1
                    ) + 1e-12
                )
            )

        return tf.cast(tf.stack(labels, axis=0), tf.float32), \
            tf.convert_to_tensor(batch_walks, dtype=tf.int32), \
            tf.stack(adjacency, axis=0), tf.stack(degree, axis=0)

    def count_nodes_within_n_hops(self, max_hops):
        # Initialize a dictionary of dictionaries to store the histograms
        histograms = {i: defaultdict(int) for i in range(max_hops + 1)}

        for start_node in self.all_relations:

            # Use a deque to implement the breadth-first search
            queue = deque([(start_node, 0)])

            # Keep track of visited nodes to avoid cycles
            visited = set([start_node])

            while queue:
                current_node, current_hop = queue.popleft()

                # Stop if we have reached the maximum number of hops
                if current_hop > max_hops:
                    break

                # Count the number of nodes within the current hop distance
                histograms[current_hop][len(visited)] += 1

                # Add all neighbors of the current node to the queue
                for neighbor in self.all_relations[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, current_hop + 1))
                        visited.add(neighbor)

        return histograms

    def plot_nodes_within_n_hops(self):
        max_hops = 8
        bin_width = 32
        histograms = self.count_nodes_within_n_hops(max_hops)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(10, 15))

        for i in range(2, max_hops):
            # Convert the histogram to a list of counts, sorted by number of 
            # reachable nodes
            counts = [histograms[i].get(j, 0) for j in
                      range(0, max(histograms[i]), bin_width)]

            ax = axes[(i - 2) // 2, (i - 2) % 2]
            ax.hist(counts, bins=range(0, max(counts) + bin_width, bin_width))

            ax.set_xlabel('Reachable vertices from start vertex')
            if (i - 2) % 2 == 0:
                ax.set_ylabel('Number of start vertices')
            ax.set_title(f'Unique vertices within {i} hops of each vertex')

            # Set the same x and y axis limits for all subplots
            # ax.set_xlim([0, global_max_x])
            # ax.set_ylim([0, global_max_y])

        plt.tight_layout()
        filepath = f'{self.config.figures_dir}/{self.config.model_name}/within_n_hop_plots.png'
        plt.savefig(filepath)

    def count_nodes_within_n_hops_of_hpo(self, max_hops):
        # Initialize a dictionary of dictionaries to store the histograms
        histograms = {i: defaultdict(int) for i in range(max_hops + 1)}

        for start_node in self.all_relations:
            # Only start at nodes that begin with 'HP:'
            if not self.index2concept[start_node].startswith('HP:'):
                continue

            # Use a deque to implement the breadth-first search
            queue = deque([(start_node, 0)])

            # Keep track of visited nodes to avoid cycles
            visited = set([start_node])

            while queue:
                current_node, current_hop = queue.popleft()

                # Stop if we have reached the maximum number of hops
                if current_hop > max_hops:
                    break

                # Count the number of nodes within the current hop distance
                histograms[current_hop][len(visited)] += 1

                # Add all neighbors of the current node to the queue
                for neighbor in self.all_relations[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, current_hop + 1))
                        visited.add(neighbor)

        return histograms

    def plot_nodes_within_n_hops_of_hpo(self):
        max_hops = 8
        bin_width = 32
        histograms = self.count_nodes_within_n_hops_of_hpo(max_hops)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(10, 15))

        for i in range(2, max_hops):
            # Convert the histogram to a list of counts, sorted by number of 
            # reachable nodes
            counts = [histograms[i].get(j, 0) for j in
                      range(0, max(histograms[i]), bin_width)]

            ax = axes[(i - 2) // 2, (i - 2) % 2]
            ax.hist(counts, bins=range(0, max(counts) + bin_width, bin_width))

            ax.set_xlabel('Reachable vertices from start vertex')
            if (i - 2) % 2 == 0:
                ax.set_ylabel('Number of start vertices')
            ax.set_title(f'Unique vertices within {i} hops of each HPO vertex')

        plt.tight_layout()
        filepath = f'{self.config.figures_dir}/{self.config.model_name}/within_n_hop_of_hpo_plots.png'
        plt.savefig(filepath)
