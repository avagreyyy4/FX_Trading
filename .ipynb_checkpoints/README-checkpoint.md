## Attempt #1:

n = 3
profit_taking = 0.005
p = 0.60

*model performs relativley well with how noisy the information is. 76% testing accuracy
* main issue is the p value --> model is not very confident with its predictions so very little entering of trade
*When p is decreased model has net negative profit - choosing to trade on bad days
NEXT STEPS:
1) decrease profit_taking value --> try to gurantee profit even if super small


NOTES
- especially important to find a optimal profit taking level to avoid imbalanced class for target value

- Goal 1: find a profit taking value that balances the classes the most - this is a bad idea actually because when taking level is roughly equal to the average std of the data then the target value can be made up by a lot of noise - natural changes rather than real patterns

- 

- Goal 2: play with n = value to add or take away model complexity