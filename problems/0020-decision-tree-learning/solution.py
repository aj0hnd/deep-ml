import math
from collections import Counter, defaultdict

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    """Calculate the information gain of splitting on attr."""
    n_example = len(labels)
    counter = Counter(labels)
    
    entropy = 0.0
    for freq in counter.values():
        prob = freq / n_example
        entropy -= prob * math.log2(prob)
    return entropy


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    labels = [example[target_attr] for example in examples]
    
    counter = dict(Counter(labels))
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    # Your code here
    labels = [example[target_attr] for example in examples]
    
    counter = dict(Counter(labels))
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # Your code here
    if len(examples) == 0:
        return 
    
    if len(set(example[target_attr] for example in examples)) == 1 or attributes == []:
        return majority_class(examples, target_attr)

    n_examples = len(examples)
    H_origin = calculate_entropy([1 if example[target_attr] == 'Yes' else 0 for example in examples])

    info_gains = [H_origin for _ in range(len(attributes))]
    for i, attr in enumerate(attributes):
        attr_split = defaultdict(list)
        for example in examples:
            attr_split[example[attr]].append(1 if example[target_attr] == 'Yes' else 0)
        for labels in attr_split.values():
            info_gains[i] -= (len(labels) / n_examples) * calculate_entropy(labels)

    selected_attr = attributes[info_gains.index(max(info_gains))]
    attributes.remove(selected_attr)
    attr_values = sorted(set(example[selected_attr] for example in examples))
    
    result = {selected_attr: {}}
    for attr_val in attr_values:
        sub_examples = [example for example in examples if example[selected_attr] == attr_val]
        subtree = learn_decision_tree(sub_examples, attributes[:], target_attr)
        if subtree is not None:
            result[selected_attr][attr_val] = subtree
    return result