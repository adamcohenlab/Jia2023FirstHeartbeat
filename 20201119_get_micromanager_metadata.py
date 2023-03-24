#! /usr/bin/python3
import json
import os
import pandas as pd

j = json.load(open("metadata.txt", "r"))
print(j.keys())
times = []
for k in list(j.keys())[1:]:
    times.append(j[k]["ElapsedTime-ms"])
df = pd.DataFrame(times, columns=["t"])
df.to_csv("E1_times.csv")