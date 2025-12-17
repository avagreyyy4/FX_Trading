## Attempt #1:

current optimal for Neural Network:
n = 5
profit_taking = 0.0035
p = 0.65

Finding balance between model accuracy and true income-based performance. Making profit taking super low caused too drastic class imbalance and forced model to be overly confident, making it extremely hard for the model to learn real signals and beat a baseline.

Increasing profit_taking to true monetary values helped distinguish the model from baseline, along with yield true profitable insights. 

Dealing a lot with raw model accuracy and model confidence/relevant application. For example, don't expect the model to have high accuracy, but should still tune to not overfit since its probability plays a big role in true monetary outcomes.