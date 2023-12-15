from niaarm import Dataset, Feature, Rule

# load the heart dataset
data = Dataset("datasets/heart.csv")

# making the rule All Features => Target([0, 1]) for our heart data
antecedent = [
    Feature("age", dtype="int", min_val=29, max_val=77),
    Feature("sex", dtype="int", min_val=0, max_val=1),
    Feature("cp", dtype="int", min_val=0, max_val=3),
    Feature("trestbps", dtype="int", min_val=94, max_val=200),
    Feature("chol", dtype="int", min_val=126, max_val=564),
    Feature("fbs", dtype="int", min_val=0, max_val=1),
    Feature("restecg", dtype="int", min_val=0, max_val=2),
    Feature("thalach", dtype="int", min_val=71, max_val=202),
    Feature("exang", dtype="int", min_val=0, max_val=1),
    Feature("oldpeak", dtype="float", min_val=0.0, max_val=6.2),
    Feature("slope", dtype="int", min_val=0, max_val=2),
    Feature("ca",  dtype="int", min_val=0, max_val=4),
    Feature("thal", dtype="int", min_val=0, max_val=3)
]
consequent = [Feature("target", dtype="int", min_val=0, max_val=1)]

# pass the transaction data to the Rule constructor to enable the calculation of metrics
rule = Rule(antecedent, consequent, transactions=data.transactions)

print(rule)
print(f"Support: {rule.support}")
print(f"Confidence: {rule.confidence}")
print(f"Lift: {rule.lift}")
