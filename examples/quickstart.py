from orthoshift import dml_ate, make_dataset

rows = make_dataset(n=300, seed=4, shift=0.5)
print(f"dml_ate={dml_ate(rows):.3f}")
