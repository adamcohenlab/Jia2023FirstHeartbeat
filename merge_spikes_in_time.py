#! /usr/bin/python3
import numpy as np
import pandas as pd
import scipy.stats as stats
import argparse
import os
import spikecounter.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument("datatable_path", help="datatable path")
parser.add_argument("spikedir")
parser.add_argument("output_folder")
parser.add_argument("geom_trace_output_folder")
args = parser.parse_args()

data = pd.read_csv(args.datatable_path, index_col="spike")
data = data.reset_index()
data["spike"] = data["spike"].astype(int)
data = data.set_index(["starttime", "t_index", "spike"])
data["t_end"] = data["t"]

starttimes = list(data.index.unique("starttime"))
vid_lookup = dict(zip(starttimes, [1,2,3,5]))

subdirs = ["E3_heart%d_preprocessed" % e for e in [1,2,3,5]]
utils.make_output_folder(args.geom_trace_output_folder)
utils.write_subfolders(args.geom_trace_output_folder, subdirs)

distance_threshold = 8
r_threshold = 0.75

merged_spikes = []

for idx, starttime in enumerate(data.index.unique("starttime")):
    linked_spikes = set([])

    vid_data = data.loc[starttime]
    vid_data = vid_data.reset_index().set_index(["t_index", "spike"])
    vid_index = vid_lookup[starttime]
    timepoints = sorted(list(vid_data.index.unique("t_index")))
    for t_idx in range(len(timepoints)-1):
        for s1 in vid_data.loc[timepoints[t_idx]].index.unique("spike"):
            if s1 not in linked_spikes:
                first_index = s1
                curr_record = vid_data.loc[timepoints[t_idx], s1].copy()
                curr_record_geom_trace = [(curr_record["x"], curr_record["y"], curr_record["volume"])]
                linked_spikes.add(s1)
                next_link_found = True
                delta_t = 0
                while next_link_found:
                    next_link_found = False
                    if t_idx + delta_t + 1 < len(timepoints):
                        try:
                            t1 = timepoints[t_idx+delta_t]
                            t2 = timepoints[t_idx+delta_t+1]
                            for s2 in vid_data.loc[t2].index.unique("spike"):
                                dist = ((vid_data.loc[t1, s1]["x"] - vid_data.loc[t2,s2]["x"])**2 + (vid_data.loc[t1, s1]["y"] - vid_data.loc[t2, s2]["y"])**2)**0.5
                                if dist < distance_threshold:
                                    trace1 = pd.read_csv("%s/E3_heart%d_preprocessed/trace%d.csv" % (args.spikedir, vid_index, s1))
                                    trace2 = pd.read_csv("%s/E3_heart%d_preprocessed/trace%d.csv" % (args.spikedir, vid_index, s2))
                                    r, _ = stats.pearsonr(trace1["intensity"], trace2["intensity"])
                                    if r > r_threshold:
                                        linked_spikes.add(s2)
                                        curr_record["dF"] = max(curr_record["dF"], vid_data.loc[t2, s2]["dF"])
                                        curr_record["intensity_mean"] = max(curr_record["intensity_mean"], vid_data.loc[t2, s2]["intensity_mean"])
                                        curr_record["intensity_max"] = max(curr_record["intensity_max"], vid_data.loc[t2, s2]["intensity_max"])
                                        curr_record["t_end"] = vid_data.loc[t2, s2]["t"]
                                        curr_record_geom_trace.append((vid_data.loc[t2, s2]["x"], vid_data.loc[t2, s2]["y"], vid_data.loc[t2, s2]["volume"]))
                                        delta_t+=1
                                        next_link_found = True
                                        print("%d linked to %d" % (s1, s2))
                                        s1 = s2
                                        break
                        except Exception:
                            print(t_idx+delta_t)
                            print(len(timepoints))
                            quit()
                curr_record["n_frames"] = delta_t+1
                curr_record["starttime"] = starttime
                merged_spikes.append(curr_record)
                geom_data = pd.DataFrame(curr_record_geom_trace, columns=["x", "y", "volume"])
                geom_data.to_csv(os.path.join(args.geom_trace_output_folder, "E3_heart%d_preprocessed" % vid_index, "spike%d_geom_trace.csv") % first_index)
merged_spikes = pd.concat(merged_spikes, axis=1).T.reset_index()
merged_spikes = merged_spikes.rename(columns={"level_0": "start_frame", "level_1": "start_spike_id"})
merged_spikes.to_csv(os.path.join(args.output_folder, "merged_spikes.csv"), index=False)