# Source: pandas 2.x "10 minutes to pandas" (https://pandas.pydata.org/docs/user_guide/10min.html): the Grouping
#   section (df with A foo/bar, B one/two/three, C and D random; groupby("A")[["C", "D"]].sum() and
#   groupby(["A", "B"]).sum()), the Merge/join section (left/right on "key"), the Reshaping pivot_table example
#   (A one/one/two/three x3, B A/B/C x4, C foo/foo/foo/bar/bar/bar x2; pivot_table(values="D", index=["A", "B"],
#   columns=["C"])) and describe() of the 6 x 4 random frame; plus the user guide "Merge, join, concatenate and compare"
#   examples (https://pandas.pydata.org/docs/user_guide/merging.html): the left join on [key1, key2] and the outer
#   join with indicator=True.
# The docs draw the random columns with np.random.randn; here they come from np.random.default_rng(2024) (same
# standard normal draws in a reproducible stream: lmpy's rng(2024) is bit-identical).
# Prints the answers as one JSON object; examples/python/pandas-groupby-merge.lms pins them.
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)

# Grouping
df = pd.DataFrame({
    "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
    "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
    "C": rng.standard_normal(8),
    "D": rng.standard_normal(8),
})
g1 = df.groupby("A")[["C", "D"]].sum()
g2 = df.groupby(["A", "B"]).sum()
g3 = df.groupby("A").agg({"C": ["mean", "std", "min", "max"], "D": ["count", "median"]})

# Merge (10 minutes)
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
m1 = pd.merge(left, right, on="key")
left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})
m2 = pd.merge(left, right, on="key")

# Merge (user guide): left join on two keys; outer join with indicator
left = pd.DataFrame({"key1": ["K0", "K0", "K1", "K2"], "key2": ["K0", "K1", "K0", "K1"],
                     "A": ["A0", "A1", "A2", "A3"], "B": ["B0", "B1", "B2", "B3"]})
right = pd.DataFrame({"key1": ["K0", "K1", "K1", "K2"], "key2": ["K0", "K0", "K0", "K0"],
                      "C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]})
lj = pd.merge(left, right, how="left", on=["key1", "key2"])
df1 = pd.DataFrame({"col1": [0, 1], "col_left": ["a", "b"]})
df2 = pd.DataFrame({"col1": [1, 2, 2], "col_right": [2, 2, 2]})
oj = pd.merge(df1, df2, on="col1", how="outer", indicator=True)

# Reshaping: pivot_table
pdf = pd.DataFrame({
    "A": ["one", "one", "two", "three"] * 3,
    "B": ["A", "B", "C"] * 4,
    "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
    "D": rng.standard_normal(12),
    "E": rng.standard_normal(12),
})
pt = pd.pivot_table(pdf, values="D", index=["A", "B"], columns=["C"])

# describe
dates = pd.date_range("20130101", periods=6)
ddf = pd.DataFrame(rng.standard_normal((6, 4)), index=dates, columns=list("ABCD"))
desc = ddf.describe()

na = lambda s: ["" if pd.isna(x) else x for x in s]  # missing text as "" (lmpy Frame.text spelling)
print(json.dumps({
    "group_a_keys": list(g1.index), "group_a_C_sum": g1["C"].tolist(), "group_a_D_sum": g1["D"].tolist(),
    "group_ab_keys": [f"{a}|{b}" for a, b in g2.index], "group_ab_C_sum": g2["C"].tolist(), "group_ab_D_sum": g2["D"].tolist(),
    "group_a_C_mean": g3[("C", "mean")].tolist(), "group_a_C_std": g3[("C", "std")].tolist(),
    "group_a_C_min": g3[("C", "min")].tolist(), "group_a_C_max": g3[("C", "max")].tolist(),
    "group_a_D_count": g3[("D", "count")].tolist(), "group_a_D_median": g3[("D", "median")].tolist(),
    "merge_same_key_rows": len(m1), "merge_same_key_lval": m1["lval"].tolist(), "merge_same_key_rval": m1["rval"].tolist(),
    "merge_key": m2["key"].tolist(), "merge_lval": m2["lval"].tolist(), "merge_rval": m2["rval"].tolist(),
    "left_join_key1": lj["key1"].tolist(), "left_join_key2": lj["key2"].tolist(), "left_join_A": lj["A"].tolist(),
    "left_join_C": na(lj["C"]), "left_join_D": na(lj["D"]),
    "outer_col1": oj["col1"].tolist(), "outer_col_left": na(oj["col_left"]), "outer_col_right": oj["col_right"].tolist(),
    "outer_merge": [str(x) for x in oj["_merge"]],
    "pivot_index": [f"{a}|{b}" for a, b in pt.index], "pivot_columns": list(pt.columns),
    "pivot_bar": pt["bar"].tolist(), "pivot_foo": pt["foo"].tolist(),
    "describe_stats": list(desc.index), "describe": {c: desc[c].tolist() for c in "ABCD"},
}))
