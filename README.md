# Building-Decision-Tree-Manually
Manually building decision tree from scratch helps to get a deeper understanding of what's hapenning under the hood when we use off-the-shelf libraries.

# What is Decision Tree?
It's a binary tree that contains split conditions inside the **nodes**, and predictions inside the **leafs**.

**Node:** The condition for branching.
**Leaf:** A node that does not split further and contains the prediction value/class.

## Information Gain
It is a method that helps to evaluate the quality of the split in the decision tree. Based on the information gain, we can choose the optimal data split at each node of the decision tree. In simple terms, it helps to ensure that each branch of the tree divides the data in the most meaningful way.

For example, in Regression task, the information gain can be measured using MSE (Mean Squared Error). We calculate MSE before the split and after. The split that results in the largest MSE reduction is considered to be the optimal choice. 

On the other hand, Weighted MSE can be better for splitting than MSE since it helps to achieve more balanced split of the node. It reduces the overall error across both splits, rather than significantly reducing error at one split, while having high error at another.

# Files
This repo contains the following files:
- **single_split.py:** calculates the best splitting threshold for a single feature
- **best_split.py:** finds the index of a feature and the best threshold for split.
- **final_tree:**
  - class Node: each node/leaf of the tree is the instance of the class Node
  - class DecisionTreeRegressor: contains methods to find the best split of the nodes using depth-wise method (splitting the nodes until one of the stopping criteria is met. In our case it's max_depth and min_samples_split).
