import pandas as pd

# Load CSV
df = pd.read_csv('text_stats_sv_train(trans).csv')

# Get last 1000 rows where ai == 1, keeping original indices
ai_1 = df[df['ai'] == 1].iloc[-1000:]
ai_1_indices = ai_1.index

# Get last 1000 rows where ai == 0, keeping original indices
ai_0 = df[df['ai'] == 0].iloc[-1000:]
ai_0_indices = ai_0.index

# Combine while keeping original order
sample = pd.concat([ai_1, ai_0])

# Get the rest by dropping using original indices
excluded_indices = ai_1_indices.union(ai_0_indices)
rest = df.drop(index=excluded_indices)

# Save to CSVs
sample.to_csv('sample.csv', index=False)
rest.to_csv('rest.csv', index=False)