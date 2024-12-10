import os
import json
import math
from PIL import Image
from einops import einops
from torch.utils.data import Dataset, DataLoader
import ast
import torch
import pandas as pd
import numpy as np
from transformers.utils import logging
import networkx as nx

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

VG_GENOME_Spatial_Relationships = [
    "above", "at", "behind", "below", "beneath", "in", "in front of",
    "inside", "on", "on top of", "to the left of", "to the right of", "under"
]

VG_GENOME_Verbs = [
    "carrying", "covered by", "covered in", "covered with", "covering", "cutting", "eating",
    "feeding", "grazing on", "hanging on", "holding", "leaning on", "looking at",
    "lying in", "lying on", "parked on", "reflected in", "resting on", "riding", "sitting at",
    "sitting in", "sitting on", "sitting on top of", "standing by", "standing in", "standing on",
    "surrounded by", "using", "walking in", "walking on", "watching", "wearing"
]

fine_graned_dependency_dictionary = {'<bos>': 0, 'acomp': 1, 'advcl': 2, 'advmod': 3, 'amod': 4, 'appos': 5, 'aux': 6,
                                     'auxpass': 7,
                                     'cc': 8, 'ccomp': 9, 'conj': 10, 'cop': 11, 'csubj': 12, 'csubjpass': 13,
                                     'dep': 14,
                                     'det': 15, 'discourse': 16, 'dobj': 17, 'expl': 18, 'infmod': 19, 'iobj': 20,
                                     'mark': 21, 'mwe': 22, 'neg': 23, 'nn': 24, 'npadvmod': 25, 'nsubj': 26,
                                     'nsubjpass': 27, 'num': 28, 'number': 29, 'parataxis': 30, 'partmod': 31,
                                     'pcomp': 32, 'pobj': 33, 'poss': 34, 'possessive': 35, 'preconj': 36, 'predet': 37,
                                     'prep': 38, 'prt': 39, 'punct': 40, 'quantmod': 41, 'rcmod': 42, 'root': 43,
                                     'tmod': 44,
                                     'xcomp': 45}


def collate_fn(examples):
    return {
        k: torch.cat([example[k] for example in examples], dim=0) if torch.is_tensor(examples[0][k]) else [example[k]
                                                                                                           for example
                                                                                                           in examples]
        for k in list(examples[0].keys())}


def get_compositional_data_loader(image_dir_path: str,
                                  metadata_path: str,
                                  preprocess_image,
                                  tokenizer, batch_size,
                                  num_workers: int,
                                  parser: str = "roberta",
                                  n_head: int = 8,
                                  args=None) -> DataLoader:
    """
    :param image_dir_path:
    :param metadata_path:
    :param preprocess_image:
    :param tokenizer:
    :param batch_size:
    :param num_workers:
    :return:
    """

    return DataLoader(dataset=CompositionalDataset(image_dir_path=image_dir_path,
                                                   metadata_path=metadata_path,
                                                   preprocess_image=preprocess_image,
                                                   tokenizer=tokenizer,
                                                   parser=parser,
                                                   n_head=n_head,
                                                   args=args),
                      batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers)


class CompositionalDataset(Dataset):
    def __init__(self,
                 image_dir_path: str,
                 metadata_path: str,
                 preprocess_image,
                 tokenizer,
                 parser: str = "roberta",
                 n_head: int = 8,
                 args=None):
        """
        :param image_dir_path: directory of image path
        :param metadata_path: path of metadata
        :param preprocess_image: preprocess image pipeline
        :param tokenizer: preprocess caption pipeline
        """

        self.image_dir_path = image_dir_path
        self.preprocess_image = preprocess_image
        self.tokenizer = tokenizer
        self.n_head = n_head
        self.args = args
        self.compute_attention_mask = compute_binary_attention

        self.context_length = self.tokenizer["tokenizer"].context_length \
            if isinstance(self.tokenizer, dict) else self.tokenizer.context_length
        self.all_special_ids = self.tokenizer["tokenizer"].all_special_ids \
            if isinstance(self.tokenizer, dict) else self.tokenizer.all_special_ids

        self.metadata_path = metadata_path
        self.non_order_compositionality_type = "vg_relation" in self.metadata_path \
                                               or "vg_attribution" in self.metadata_path \
                                               or "sugar_crepe" in self.metadata_path \
                                               or "colorswap" in self.metadata_path \
                                               or "vl-checklist" in self.metadata_path
        self.parser = parser
        self.parse_tree = self.parse_dependency_tree
        self.dependency_dictionary = fine_graned_dependency_dictionary

        if self.metadata_path[-4:] in ["json"]:
            with open(metadata_path, 'r') as metadata_file:
                self.annotation = json.load(metadata_file)
        else:
            self.annotation = pd.read_csv(metadata_path, sep='\t')

        if "visual_genome_attribution" in self.metadata_path:
            self.all_attributes_or_relations = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item
                                                in self.annotation]

        elif any(map(self.metadata_path.__contains__, ["visual_genome_relation", "sugar_crepe", "vl-checklist",
                                                       "fg-ovd"])):
            self.all_attributes_or_relations = [item["relation_name"] for item in self.annotation]

        self.dict_embeddings = {"gt_images": [],
                                "captions": [],
                                "gt_input_ids": [],
                                "false_input_ids": []}

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        test_case = self.annotation[index]

        # Open image
        image = Image.open(os.path.join(self.image_dir_path, test_case["image_path"])).convert('RGB') if isinstance(
            test_case["image_path"], str) else [
            Image.open(os.path.join(self.image_dir_path, test_case["image_path"][i])).convert('RGB')
            for i in range(len(test_case["image_path"]))]

        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        if "bbox_x" in list(test_case.keys()):
            image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"],
                                test_case["bbox_y"] + test_case["bbox_h"]))

        if "bbox" in list(test_case.keys()):
            x, y, w, h = int(np.floor(test_case['bbox'][0])), int(np.floor(test_case['bbox'][1])), \
                         int(np.ceil(test_case['bbox'][2])), int(np.ceil(test_case['bbox'][3]))
            image = image.crop((x, y, x + w, y + h))

        if self.args.instruct_blip:
            image = self.transform_image_instruct_blip(image)
        else:
            image = self.transform_image(image).unsqueeze(dim=0) if isinstance(test_case["image_path"], str) else \
                torch.cat([self.transform_image(img).unsqueeze(dim=0) for img in image], dim=0)

        # Each test case has a correct and incorrect caption.
        true_captions = test_case["true_caption"]
        false_captions = test_case["false_caption"]

        if "fg-ovd" in self.metadata_path:
            while len(false_captions) < 10:
                false_captions.append(false_captions[0])
                test_case["false_caption_tree_crf"].append(test_case["false_caption_tree_crf"][0])
                test_case["false_caption_tree_biaffine"].append(test_case["false_caption_tree_biaffine"][0])
                test_case["false_caption_tree_roberta"].append(test_case["false_caption_tree_roberta"][0])

        # Tokenizer captions
        false_input_ids = self.transform_caption(false_captions)

        # Parse the dependency graph
        true_caption_tree_crf = test_case["caption_tree_crf"]
        true_caption_tree_biaffine = test_case["caption_tree_biaffine"]
        true_caption_tree_roberta = test_case["caption_tree_roberta"]

        agreement = True
        true_caption_tree = self.read_tree(caption_tree_roberta=true_caption_tree_roberta,
                                           caption_tree_crf=true_caption_tree_crf,
                                           caption_tree_biaffine=true_caption_tree_biaffine)
        gt_input_ids = self.transform_caption(true_captions, tree=true_caption_tree)

        false_caption_trees_crf = test_case["false_caption_tree_crf"]
        false_caption_trees_biaffine = test_case["false_caption_tree_biaffine"]
        false_caption_trees_roberta = test_case["false_caption_tree_roberta"]

        false_caption_masks = []
        false_likelihood_ids = []

        if not self.non_order_compositionality_type or "vl-checklist" in self.metadata_path:
            false_caption_trees = []
            for i in range(len(false_caption_trees_crf)):
                false_caption_tree = self.read_tree(caption_tree_roberta=false_caption_trees_roberta,
                                                    caption_tree_crf=false_caption_trees_crf,
                                                    caption_tree_biaffine=false_caption_trees_biaffine,
                                                    i=i)
                false_input_ids[i] = self.transform_caption(false_captions[i], tree=false_caption_tree)

                false_caption_trees.append(false_caption_tree)

            true_caption_mask, gt_likelihood_ids = self.parse_tree_to_attention_mask(true_caption_tree)
            for i, false_caption_tree in enumerate(false_caption_trees):
                false_cap_mask, false_id = self.parse_tree_to_attention_mask(false_caption_tree)
                false_caption_masks.append(false_cap_mask)
                false_likelihood_ids.append(false_id)

            false_caption_mask = torch.cat(false_caption_masks, dim=0)
            false_likelihood_ids = torch.cat(false_likelihood_ids, dim=0)
        else:
            false_caption_tree = self.read_tree(caption_tree_roberta=false_caption_trees_roberta,
                                                caption_tree_crf=false_caption_trees_crf,
                                                caption_tree_biaffine=false_caption_trees_biaffine)
            false_input_ids = self.transform_caption(false_captions, tree=false_caption_tree)

            true_caption_mask, gt_likelihood_ids = self.parse_tree_to_attention_mask(true_caption_tree)
            false_caption_mask, false_likelihood_ids = self.parse_tree_to_attention_mask(false_caption_tree)

        return dict({"gt_images": image,
                     "gt_input_ids": gt_input_ids,
                     "false_input_ids": false_input_ids,
                     "true_caption_mask": true_caption_mask,
                     "false_caption_mask": false_caption_mask,
                     "gt_likelihood_ids": gt_likelihood_ids,
                     "false_likelihood_ids": false_likelihood_ids,
                     "raw_caption": true_captions,
                     "raw_caption_false": false_captions})

    def transform_image(self, image: Image) -> torch.Tensor:
        """
        :param image:
        :return:
        """
        return self.preprocess_image(image)

    def transform_image_instruct_blip(self, image) -> torch.Tensor:
        """
        :param image:
        :return:
        """

        image = torch.Tensor(np.array(self.transform_image(image)["pixel_values"]))
        image = torch.cat([torch.Tensor(image)])

        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)

        return image

    def transform_caption(self, caption: str, tree=None) -> torch.Tensor:

        # Tokenize

        if tree in [None]:
            caption = self.tokenizer["tokenizer"](caption)

            # Exclude BOS and last token
            caption = caption[:, 1:-1]

            # Create sequence of mask token (<MASK> = BOS)
            pre_caption = torch.zeros(caption.shape[0], caption.shape[1]) + \
                          self.tokenizer["tokenizer"].all_special_ids[0]

            # Cat caption
            return torch.cat([pre_caption, caption], dim=1).to(torch.int64)
        else:
            # Case parser: tokenize from the tokenization of the parser
            caption = [element[0] for element in tree]
            caption = self.tokenizer["tokenizer"](caption)

            post_caption = [caption[i][j].item() for i in range(caption.shape[0])
                            for j in range(caption.shape[1]) if caption[i][j].item()
                            not in self.all_special_ids + [0]][:self.context_length - 2]
            post_caption = post_caption + [0 for _ in range(self.context_length - 2 - len(post_caption))]
            post_caption = torch.Tensor(post_caption).unsqueeze(dim=0)

            # Case use dependency mask token
            pre_caption = []
            # Cycle over word
            for i in range(caption.shape[0]):
                tokens = []
                # Cycle over tokens (a word in tree can be represented by multiple tokens)
                for j in range(caption.shape[1]):
                    # Append real token to the list
                    if caption[i][j].item() not in self.all_special_ids + [0]:
                        tokens.append(caption[i][j].item())

                if len(tokens) != 0:
                    pre_caption = pre_caption + [self.all_special_ids[0] + self.dependency_dictionary[tree[i][2]]]
                # If the word is represented by multiple token, append the <MSK_MULTIPLE_TOKEN> for len(tokens) - 1
                if len(tokens) > 1:
                    pre_caption = pre_caption + [len(set(self.dependency_dictionary.values())) +
                                                 self.all_special_ids[0]] * (len(tokens) - 1)
            # Cast pre_caption to max_length
            pre_caption = pre_caption[:self.context_length - 2]
            # Fill with zeros
            pre_caption = pre_caption + [0 for _ in range(self.context_length - 2 - len(pre_caption))]
            # Transform to Tensor
            pre_caption = torch.Tensor(pre_caption).unsqueeze(dim=0)

            # Cat caption
            return torch.cat([pre_caption, post_caption], dim=1).to(torch.int64)

    def update_embeddings(self, dict_embeddings: dict) -> None:
        """
        Store embedding information for metric computation. In autoregressive mode, only positive-negative logits
        are stored directly in the embedding dict

        :param dict_embeddings: it stores batch embedding
        """
        for k in list(dict_embeddings.keys()):
            self.dict_embeddings[k].append(dict_embeddings[k])

    def clean_embedding_dict(self) -> None:
        self.dict_embeddings = {k: [] for k in list(self.dict_embeddings.keys())}

    def cat_embeddings(self):
        # Concatenate images and captions options
        # image_options: B x K x D (K is always 1 for ARO)
        # caption_options: B x L x D (L is the concatenation of true and false captions)
        image_options = np.concatenate(self.dict_embeddings["gt_images"], axis=0)
        caption_options = np.concatenate(self.dict_embeddings["captions"], axis=0)

        return image_options, caption_options

    def compute_metrics(self) -> dict:
        """

        :return: dict with metrics
        """

        # Initialize metrics dict
        metrics = {}
        image_options, caption_options = self.cat_embeddings()

        # Compute the scores matrix
        scores = np.concatenate((image_options, caption_options), axis=1)

        # Predictions are indices of maximum scores along the L dimension
        preds = np.argmax(scores, axis=-1)

        # The 0 index occurs when the true caption is most similar to the selected image
        correct_mask = (preds == 0)

        # The average number of 0s of the argmax represents the accuracy
        metric_value = np.mean(correct_mask)

        # If the dataset is visual genome relation or attribution
        if any(map(self.metadata_path.__contains__, ["visual_genome_relation",
                                                     "visual_genome_attribution",
                                                     "sugar_crepe",
                                                     "vl-checklist",
                                                     "fg-ovd"])):
            # Initialize fine grained metric dict
            metrics.update({"fine_grained_results": {}})

            # Set counter to compute avg accuracy
            attributes_or_relations_counter = 0
            attributes_or_relations_accuracy = 0

            # Compute the set of attribution or relation uniques
            all_attributes_or_relations = np.array(self.all_attributes_or_relations)

            # Compute set of uniques attribution or relations
            unique_attribution_or_relations = set(all_attributes_or_relations)

            # VG Relation case: filter relation and verbs
            if any(map(self.metadata_path.__contains__, ["visual_genome_relation"])):
                unique_attribution_or_relations = unique_attribution_or_relations. \
                    intersection(set(VG_GENOME_Verbs).union(set(VG_GENOME_Spatial_Relationships)))

            # Initialize variables to compute the weighted avg accuracy in VG
            correct_predictions = 0
            items = 0

            # For each attribution/relation
            for attr_or_rel in unique_attribution_or_relations:
                attr_or_rel_mask = (all_attributes_or_relations == attr_or_rel)
                if "visual_genome_relation" in self.metadata_path:
                    if attr_or_rel_mask.sum() == 0:
                        continue
                elif "visual_genome_attribution" in self.metadata_path:
                    if attr_or_rel_mask.sum() < 25:
                        continue

                # Avg accuracy per item
                attr_or_rel_accuracy = correct_mask[attr_or_rel_mask[:correct_mask.shape[0]]].mean()

                # Number of items considered for the current relation/attribution
                items += attr_or_rel_mask.sum()

                # Number of items correctly classified
                correct_predictions += correct_mask[attr_or_rel_mask[:correct_mask.shape[0]]].sum()

                # Update variables to compute the avg accuracy
                attributes_or_relations_accuracy += attr_or_rel_accuracy
                attributes_or_relations_counter += 1

                # Update the metric dict with the accuracy of the current item
                metrics["fine_grained_results"].update({attr_or_rel: attr_or_rel_accuracy})

            # Compute the weighted avg accuracy
            metric_value = correct_predictions / items if items != 0 else 0

            # Compute avg accuracy
            avg_accuracy = attributes_or_relations_accuracy / attributes_or_relations_counter \
                if attributes_or_relations_counter > 0 else 0

            # Update the metric dict with the avg accuracy
            metrics.update({"avg_accuracy": avg_accuracy})

        # Compute metric string based on number of false captions and update metrics computed
        metric_string = "top_1_recall_img2text" if caption_options.shape[1] > 2 else "accuracy"

        # Update metric dict
        metrics.update({metric_string: metric_value})

        return metrics

    def compute_metrics_colorswap(self) -> dict:

        # Initialize metrics dict
        metrics = {}
        image_options, caption_options = self.cat_embeddings()
        scores = np.concatenate((image_options, caption_options), axis=1)

        c0_i0 = scores[::2, 0]
        c0_i1 = scores[1::2, 0]
        c1_i0 = scores[::2, 1]
        c1_i1 = scores[1::2, 1]
        text_score = np.logical_and(c0_i0 > c1_i0, c1_i1 > c0_i1)

        metrics.update({"text_score": np.mean(text_score)})

        return metrics

    def expand_tree(self, tree, ids_per_token):

        # Offsets stores the index of element with multiple tokens
        offsets = []

        # The expanded tree: each word will correspond to 1 and only 1 token
        new_tree = []

        # Update offset list
        for i in range(len(ids_per_token)):
            for j in range(len(ids_per_token[i]) - 1):
                offsets.append(i + 1)

        # For each element of the tree
        for i in range(len(ids_per_token)):
            # If the word is represented by only one token: update tree with index updated according to offset
            if len(ids_per_token[i]) == 1:
                new_tree.append([tree[i][0], tree[i][1] + sum([tree[i][1] >= offsets[j]
                                                               for j in range(len(offsets))])])
            elif len(ids_per_token[i]) != 0:
                # Case word represented by multiple tokens
                # For each element until the last - 1 token

                # The fist token of the current word represented by multiple tokens, contains the dependency
                # of the original word (the offset list must be considered)
                new_tree.append([self.tokenizer["tokenizer"].decoder[ids_per_token[i][0]],
                                 tree[i][1] + sum([tree[i][1] >= offsets[j] for j in range(len(offsets))])])

                for k in range(1, len(ids_per_token[i])):
                    # A word composed by multiple tokens has the next token as dependency:
                    # Example:
                    # INPUT: ["a", 2], ["snowboarder", 0] -> ["a", 2], ["snowboard", 3], ["er", 0]
                    new_tree.append([self.tokenizer["tokenizer"].decoder[ids_per_token[i][k]], len(new_tree)])

        return new_tree, [x for xs in ids_per_token for x in xs]

    def parse_tree_to_attention_mask(self, tree):
        # Compute ids of each token of the sentence separately (this is used to align the CLIP tokenizer with the
        # tokenizer used to compute the tree

        sentence = [element[0] for element in tree]

        separated_ids = self.tokenizer["tokenizer"](sentence)

        # Compute the id which is not the start or end token from the separate_id previously tokenized set of words
        ids_per_token = [[separated_ids[i][j].item()
                          for j in range(separated_ids.shape[1]) if separated_ids[i][j].item() != 0
                          and separated_ids[i][j].item() not in self.all_special_ids]
                         for i in range(separated_ids.shape[0])]

        new_tree, flattened_ids_per_token = self.expand_tree(tree, ids_per_token)

        # Check if tree is correct
        # Initialize the dict edges: key = index of each separated word; ids_per_token: list of tokens representing
        # the i-th word in the sentence (the i-th token has an edge with itself)
        dict_edges = {i: [flattened_ids_per_token[i], [], None] for i in range(len(new_tree))}

        # Parse tree according the specified modality
        processed_dict = self.parse_tree(new_tree, dict_edges)

        # Compute attention mask
        attn_mask = self.compute_attention_mask(processed_dict, (self.context_length - 2) * 2, self.n_head)

        ground_truth = flattened_ids_per_token[:self.context_length - 2] + [0 for _ in
                                                                            range(attn_mask.shape[2] - len(
                                                                                flattened_ids_per_token[
                                                                                :self.context_length - 2]))]
        ground_truth = torch.Tensor(ground_truth).unsqueeze(dim=0)

        return attn_mask, ground_truth

    def parse_dependency_tree(self, tree, dict_edges) -> dict:
        """
        Generate dict with specified attention token according to dependency graph and next token in the graph structure
        """

        for i in range(len(tree)):

            # Initialize graph
            graph = {i: [] for i in range(len(tree))}

            # Populate graph
            for i in range(len(tree)):
                if tree[i][1] != 0:
                    # Append edge in graph
                    graph[tree[i][1] - 1].append(i)

            # Find all pairs of nodes such that the value is a common ancestor
            graph = nx.DiGraph(graph)

            if not nx.is_directed_acyclic_graph(graph):
                return dict_edges

            graph_dict = dict(nx.all_pairs_lowest_common_ancestor(graph))

            # Populate dict_edges: a token t makes attention with the token s if s is a direct child or child of a
            # direct child of t

            for k in list(graph_dict.keys()):
                current_node = graph_dict[k]
                if current_node not in dict_edges[k[0]][1] and current_node != k[0]:
                    dict_edges[k[0]][1].append(current_node)
                    dict_edges[k[0]][1].sort()
                if current_node not in dict_edges[k[1]][1] and current_node != k[1]:
                    dict_edges[k[1]][1].append(current_node)
                    dict_edges[k[1]][1].sort()

        return dict_edges

    def read_tree(self,
                  caption_tree_roberta: list,
                  caption_tree_crf: list,
                  caption_tree_biaffine: list,
                  i: int = None) -> list:
        """
        Read tree according to self.parser
        """
        caption_tree = None

        if self.parser in ["crf"]:
            caption_tree = caption_tree_crf
        elif self.parser in ["biaffine"]:
            caption_tree = caption_tree_biaffine
        elif self.parser in ["roberta"]:
            caption_tree = caption_tree_roberta

        if i not in [None]:
            caption_tree = caption_tree[i]

        return caption_tree


class TrainingSet(CompositionalDataset):
    def __init__(self,
                 image_dir_path: str,
                 metadata_path: str,
                 preprocess_image,
                 tokenizer,
                 parser: str = "roberta",
                 n_head: int = 8,
                 args=None):
        """
        :param image_dir_path: directory of image path
        :param metadata_path: path of metadata
        :param preprocess_image: preprocess image pipeline
        :param tokenizer: preprocess caption pipeline
        """

        super().__init__(image_dir_path=image_dir_path,
                         metadata_path=metadata_path,
                         preprocess_image=preprocess_image,
                         tokenizer=tokenizer,
                         parser=parser,
                         n_head=n_head,
                         args=args)

    def __getitem__(self, index: int) -> dict:
        # Sample element
        test_case = self.annotation.iloc[index]

        # Open image and negative images
        image = Image.open(os.path.join(self.image_dir_path, test_case["filepath"])).convert('RGB')

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["title"]

        if self.args.instruct_blip:
            image = self.transform_image_instruct_blip(image)
        else:
            # Apply image transformations
            image = self.transform_image(image).unsqueeze(dim=0)

        # Compute mask
        # Parse the dependency graph
        true_caption_tree_crf = ast.literal_eval(test_case["caption_tree_crf"])
        true_caption_tree_biaffine = ast.literal_eval(test_case["caption_tree_biaffine"])
        true_caption_tree_roberta = ast.literal_eval(test_case["caption_tree_roberta"])

        true_caption_tree = self.read_tree(caption_tree_roberta=true_caption_tree_roberta,
                                           caption_tree_crf=true_caption_tree_crf,
                                           caption_tree_biaffine=true_caption_tree_biaffine)
        gt_input_ids = self.transform_caption(true_caption, tree=true_caption_tree)

        true_caption_mask, ground_truth = self.parse_tree_to_attention_mask(true_caption_tree)

        return dict({"gt_images": image,
                     "gt_input_ids": gt_input_ids,
                     "true_caption_mask": true_caption_mask,
                     "true_caption": true_caption,
                     "raw_caption": true_caption,
                     "ground_truth": ground_truth
                     })


class ValidationSet(CompositionalDataset):
    def __init__(self,
                 image_dir_path: str,
                 metadata_path: str,
                 preprocess_image,
                 tokenizer,
                 parser: str = "crf",
                 n_head: int = 8,
                 args=None):
        """
        :param image_dir_path: directory of image path
        :param metadata_path: path of metadata
        :param preprocess_image: preprocess image pipeline
        :param tokenizer: preprocess caption pipeline
        """
        super().__init__(image_dir_path=image_dir_path,
                         metadata_path=metadata_path,
                         preprocess_image=preprocess_image,
                         tokenizer=tokenizer,
                         parser=parser,
                         n_head=n_head,
                         args=args)

        self.dict_embeddings = {"gt_input_ids": [],
                                "false_input_ids": [],
                                "unique_ids": []}

    def __getitem__(self, index: int) -> dict:

        false_input_ids = None

        # Sample element
        test_case = self.annotation.iloc[index]

        # Open image and negative images
        image = Image.open(os.path.join(self.image_dir_path, test_case["filepath"])).convert('RGB')

        if self.args.instruct_blip:
            image = self.transform_image_instruct_blip(image)
        else:
            # Apply image transformations
            image = self.transform_image(image).unsqueeze(dim=0)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["title"]
        false_caption = ast.literal_eval(test_case["neg_caption"])

        # Parse the dependency graph
        true_caption_tree_crf = ast.literal_eval(test_case["caption_tree_crf"])
        true_caption_tree_biaffine = ast.literal_eval(test_case["caption_tree_biaffine"])
        true_caption_tree_roberta = ast.literal_eval(test_case["caption_tree_roberta"])

        true_caption_tree = self.read_tree(caption_tree_roberta=true_caption_tree_roberta,
                                           caption_tree_crf=true_caption_tree_crf,
                                           caption_tree_biaffine=true_caption_tree_biaffine)
        gt_input_ids = self.transform_caption(true_caption, tree=true_caption_tree)

        false_caption_trees_crf = ast.literal_eval(test_case["false_caption_tree_crf"])
        false_caption_trees_biaffine = ast.literal_eval(test_case["false_caption_tree_biaffine"])
        false_caption_trees_roberta = ast.literal_eval(test_case["false_captions_tree_roberta"])

        false_caption_masks = []
        false_ids = []
        false_caption_trees = []

        for i in range(len(false_caption_trees_crf)):
            false_caption_tree = self.read_tree(caption_tree_roberta=false_caption_trees_roberta,
                                                caption_tree_crf=false_caption_trees_crf,
                                                caption_tree_biaffine=false_caption_trees_biaffine,
                                                i=i)
            false_input_ids[i] = self.transform_caption(false_caption[i], tree=false_caption_tree)

            false_caption_trees.append(false_caption_tree)

        true_caption_mask, gt_likelihood_ids = self.parse_tree_to_attention_mask(true_caption_tree)

        for i, false_caption_tree in enumerate(false_caption_trees):
            false_cap_mask, false_id = self.parse_tree_to_attention_mask(false_caption_tree)
            false_caption_masks.append(false_cap_mask)
            false_ids.append(false_id)

        false_caption_mask = torch.cat(false_caption_masks)
        false_likelihood_ids = torch.cat(false_ids)

        unique_ids = torch.Tensor([index for _ in range(false_input_ids.shape[0])])

        return dict({"gt_images": image,
                     "gt_input_ids": gt_input_ids,
                     "true_caption_mask": true_caption_mask,
                     "false_input_ids": false_input_ids,
                     "false_caption_mask": false_caption_mask,
                     "true_caption": true_caption,
                     "gt_likelihood_ids": gt_likelihood_ids,
                     "false_likelihood_ids": false_likelihood_ids,
                     "unique_ids": unique_ids})

    def compute_metrics(self) -> dict:
        # Concatenate embeddings and unique ids
        captions_embeddings = np.concatenate(self.dict_embeddings["gt_input_ids"], axis=0)
        false_caption_embeddings = np.concatenate(self.dict_embeddings["false_input_ids"], axis=0)
        unique_ids = np.concatenate(self.dict_embeddings["unique_ids"], axis=0)

        # Create matrix scores (6 columns because samples could have max 5 false captions)
        scores = np.array([[-math.inf for _ in range(6)] for _ in range(captions_embeddings.shape[0])])

        # Fill true caption score column
        scores[:, 0] = captions_embeddings[:]

        # Fill negative score columns
        false_items = 0
        index = 1

        for i in range(unique_ids.shape[0]):
            if unique_ids[i] != false_items:
                index = 1
                false_items = false_items + 1
            scores[int(unique_ids[i]), index] = false_caption_embeddings[i]
            index = index + 1

        hard_negative_accuracy = np.sum(scores[:, 0] > np.max(scores[:, 1:], axis=1)) / scores.shape[0]

        return {"validation_hard_negative_accuracy": hard_negative_accuracy}


def compute_binary_attention(processed_dict: dict,
                             context_length: int,
                             num_heads: int) -> torch.Tensor:
    # Create mask with -inf except on the main diagonal and the first column (each token attends the first token)
    attn_mask = torch.ones(size=(context_length, context_length)) == 0
    attn_mask.fill_diagonal_(True)

    for i in range(min(len(list(processed_dict.keys())), int(context_length / 2))):
        for j in range(len(processed_dict[i][1])):
            if processed_dict[i][1][j] < int(context_length / 2):
                attn_mask[i, processed_dict[i][1][j] + int(context_length / 2)] = True
                attn_mask[i + int(context_length / 2), processed_dict[i][1][j] + int(context_length / 2)] = True

    # Unsqueeze and repeat the mask for the number of heads
    attn_mask = attn_mask.unsqueeze(dim=0).unsqueeze(dim=0)
    attn_mask = einops.repeat(attn_mask, "1 1 H W -> 1 N H W", N=num_heads)

    return attn_mask
