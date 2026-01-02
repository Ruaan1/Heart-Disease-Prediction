import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# ---------------- Example Inference Inputs ----------------
# You can use any of the following predefined evidence dictionaries
# for testing. These are known to exist in the training data and
# will yield meaningful results from the model. Use these for the seen results.

# Example 1:
# evidence = {
#     'age_cat': 'old',
#     'chol_cat': 'high',
#     'bp_cat': 'high',
#     'thalach_cat': 'normal',
#     'oldpeak_cat': 'moderate'
# }

# Example 2:
# evidence = {
#     'age_cat': 'middle',
#     'chol_cat': 'normal',
#     'bp_cat': 'low',
#     'thalach_cat': 'high',
#     'oldpeak_cat': 'high'
# }

# Example 3:
# evidence = {
#     'age_cat': 'young',
#     'bp_cat': 'normal',
#     'chol_cat': 'high',
#     'thalach_cat': 'high',
#     'oldpeak_cat': 'high'
# }

# -----------------------------------------------------------
# You can also try your own evidence combinations, but if your input
# contains a combination not present in the training data, the model
# may return uniform probabilities (i.e., 0.2 for all targets).

df = pd.read_csv('../data/heart_disease_clean.csv')

# shows the structure and variables used
model = DiscreteBayesianNetwork([
    ('age_cat', 'target'),
    ('chol_cat', 'target'),
    ('bp_cat', 'target'),
    ('thalach_cat', 'target'),
    ('oldpeak_cat', 'target')
])

# fits model
model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')

# Sets up inference using vatiable elimination
inference = VariableElimination(model)

# Define your evidence
seen_evidence = {
    'age_cat': 'middle',
    'bp_cat': 'high',
    'chol_cat': 'normal',
    'thalach_cat': 'high',
    'oldpeak_cat': 'moderate'
}
contradicting_evidence = {
    'age_cat': 'middle',
    'bp_cat': 'high',
    'chol_cat': 'low',
    'thalach_cat': 'high',
    'oldpeak_cat': 'moderate'
}
specific_high_evidence = {
    'age_cat': 'old',
    'bp_cat': 'high',
    'chol_cat': 'high',
    'thalach_cat': 'high',
    'oldpeak_cat': 'high'
}
unseen_evidence = {
    'age_cat': 'old',
    'bp_cat': 'high',
    'chol_cat': 'high',
    'thalach_cat': 'low',
    'oldpeak_cat': 'high'
}
moderate_risk_evidence = {
    'age_cat': 'middle',         
    'bp_cat': 'normal',          
    'chol_cat': 'high',          
    'thalach_cat': 'normal',     
    'oldpeak_cat': 'moderate'    
}

# Query the probability of target given evidence
seen_posterior = inference.query(variables=['target'], evidence=seen_evidence)
contradicting_posterior = inference.query(variables=['target'], evidence=contradicting_evidence)
specific_posterior = inference.query(variables=['target'], evidence=specific_high_evidence)
unseen_posterior = inference.query(variables=['target'], evidence=unseen_evidence)
moderate_posterior = inference.query(variables=['target'], evidence=moderate_risk_evidence)

# Display result
print("\nInference result for P(target | seen_evidence):")
print(seen_posterior)

#print("\nInference result for P(target | contradicting_evidence):")
#print(contradicting_posterior)

#print("\nInference result for P(target | specific_high_evidence):")
#print(specific_posterior)

#print("\nInference result for P(target | moderate_risk_evidence):")
#print(moderate_posterior)

#print("\nInference result for P(target | unseen_evidence):")
#print(unseen_posterior)

