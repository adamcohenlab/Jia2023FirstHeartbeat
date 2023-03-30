import numpy as np
import scipy.stats as stats

np.seterr(all="raise")


def spearman_image_vs_trace(img, trace):
    correlation = np.zeros((img.shape[1], img.shape[2], img.shape[3]))
    for z in range(img.shape[1]):
        for y in range(img.shape[2]):
            for x in range(img.shape[3]):
                correlation[z, y, x] = stats.spearmanr(
                    img[:, z, y, x].astype(float), trace.astype(float)
                )[0]
                if np.isnan(correlation[z, y, x]):
                    correlation[z, y, x] = 0
    return correlation


def pearson_image_vs_trace(img, trace):
    correlation = np.zeros((img.shape[1], img.shape[2], img.shape[3]))
    for z in range(img.shape[1]):
        for y in range(img.shape[2]):
            for x in range(img.shape[3]):
                correlation[z, y, x] = stats.pearsonr(
                    img[:, z, y, x].astype(float), trace.astype(float)
                )[0]
                if np.isnan(correlation[z, y, x]):
                    correlation[z, y, x] = 0
    return correlation


def dot_image_vs_trace(img, trace):
    correlation = np.zeros((img.shape[1], img.shape[2], img.shape[3]))
    normalized_trace = trace.astype(float) - np.mean(trace.astype(float))
    normalized_trace = normalized_trace / np.linalg.norm(normalized_trace)

    for z in range(img.shape[1]):
        for y in range(img.shape[2]):
            for x in range(img.shape[3]):
                pixel_trace = img[:, z, y, x].astype(float)
                pixel_trace = pixel_trace - np.mean(pixel_trace)
                try:
                    if np.any(pixel_trace != 0):
                        pixel_trace = pixel_trace / np.linalg.norm(pixel_trace)
                except Exception:
                    print(x, y, z)
                    print(img[:, z, y, x])
                    print(pixel_trace)
                    exit()
                correlation[z, y, x] = np.dot(pixel_trace, normalized_trace)
    return correlation
