## Attempt #1:

n = 3
profit_taking = 0.005
p = 0.60

*model performs relativley well with how noisy the information is. 76% testing accuracy
* main issue is the p value --> model is not very confident with its predictions so very little entering of trade
*When p is decreased model has net negative profit - choosing to trade on bad days
NEXT STEPS:
1) decrease profit_taking value --> try to gurantee profit even if super small
