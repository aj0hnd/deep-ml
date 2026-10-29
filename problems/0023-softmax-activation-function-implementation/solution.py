import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores = [score - max(scores) for score in scores]
    Z = sum(math.exp(score) for score in scores)
    return [math.exp(score) / Z for score in scores]