import torch
import math
from collections import Counter
from typing import List, Dict, Any, Union, defaultdict


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    # Your implementation here
    n_examples = len(labels)
    counter = Counter(labels)

    entropy = 0.0
    for value in counter.values():
        prob = value / n_examples
        entropy -= (prob) * math.log2(prob)
    return entropy


def calculate_information_gain(
    examples: List[Dict[str, Any]],
    attr: str,
    target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """
    # Your implementation here
    origin_labels = [example[target_attr] for example in examples]
    info_gain = calculate_entropy(origin_labels)
    
    splited_examples = defaultdict(list)
    for example in examples:
        splited_examples[example[attr]].append(example)

    n_examples = len(examples)
    for (attr_val, sub_examples) in splited_examples.items():
        info_gain -= (len(sub_examples) / n_examples) * calculate_entropy([example[target_attr] for example in sub_examples])
    return info_gain


def majority_class(
    examples: List[Dict[str, Any]],
    target_attr: str
) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    In case of a tie, return the class that comes first alphabetically.
    """
    # Your implementation here
    counter = Counter(example[target_attr] for example in examples)
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]


def learn_decision_tree(
    examples: List[Dict[str, Any]],
    attributes: List[str],
    target_attr: str
) -> Union[Dict[str, Any], Any]:
    """
    Learn a decision tree using the ID3 algorithm.
    Returns either a nested dict representing the tree or a class label at the leaves.
    """
    # Your implementation here
    if len(examples) == 0:
        return

    if len(set(example[target_attr] for example in examples)) == 1 or len(attributes) == 0:
        return majority_class(examples, target_attr)

    selected_attr = None
    max_info_gain = -1
    for attr in attributes:
        temp_info_gain = calculate_information_gain(examples, attr, target_attr)
        if temp_info_gain > max_info_gain:
            selected_attr = attr
            max_info_gain = temp_info_gain

    sub_attributes = [attr for attr in attributes if attr != selected_attr]
    splited_examples = defaultdict(list)
    for example in examples:
        splited_examples[example[selected_attr]].append(example)

    result = {selected_attr: {}}
    for attr_val in sorted(splited_examples.keys()):
        subtree = learn_decision_tree(splited_examples[attr_val], sub_attributes, target_attr)
        if subtree is not None:
            result[selected_attr][attr_val] = subtree
    return result
