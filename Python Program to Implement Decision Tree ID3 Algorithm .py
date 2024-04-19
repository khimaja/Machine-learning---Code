import random
!pip install math
import math


def entropy(data):
  """
  Calculates the entropy of a dataset.

  Args:
    data: A list of data points.

  Returns:
    The entropy of the dataset.
  """

  p = {}
  for point in data:
    label = point[-1]
    if label not in p:
      p[label] = 0
    p[label] += 1

  entropy = 0
  for label, count in p.items():
    prob = count / len(data)
    entropy -= prob * math.log2(prob)

  return entropy

def information_gain(data, attribute):
  """
  Calculates the information gain of a dataset for a given attribute.

  Args:
    data: A list of data points.
    attribute: The attribute to calculate the information gain for.

  Returns:
    The information gain of the dataset for the given attribute.
  """

  entropy_before = entropy(data)
  values = set(point[attribute] for point in data)
  entropy_after = 0
  for value in values:
    data_filtered = [point for point in data if point[attribute] == value]
    entropy_after += entropy(data_filtered) * len(data_filtered) / len(data)

  return entropy_before - entropy_after

def id3(data, attributes):
  """
  Builds a decision tree from a dataset.

  Args:
    data: A list of data points.
    attributes: The list of attributes to consider.

  Returns:
    A decision tree.
  """

  if len(attributes) == 0:
    return data[0][-1]

  attribute = max(attributes, key=lambda attribute: information_gain(data, attribute))
  tree = {}
  for value in set(point[attribute] for point in data):
    data_filtered = [point for point in data if point[attribute] == value]
    tree[value] = id3(data_filtered, attributes[:-1])

  return tree

def main():
  data = [['Sunny', 'Warm', 'High', 'Yes'],
           ['Sunny', 'Warm', 'Normal', 'No'],
           ['Overcast', 'Cool', 'Normal', 'Yes'],
           ['Rainy', 'Cold', 'Normal', 'No'],
           ['Rainy', 'Cold', 'High', 'No'],
           ['Overcast', 'Cool', 'High', 'Yes']]
  attributes = ['Outlook', 'Temperature', 'Humidity']
  tree = id3(data, attributes)
  print(tree)

if __name__ == '__main__':
  main()
