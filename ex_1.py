import sys
import numpy as np
import scipy.spatial
import scipy.io.wavfile
import matplotlib.pyplot as plt


def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)
    f = open("output.txt", "w+")
    run = True
    i = 0
    while i < 30:
        # finding the distances to centroids for each point

        distances = scipy.spatial.distance.cdist(x, centroids)
        clusters = create_clusters(distances, x)
        new_cents = create_new_cents(clusters)
        f.write(f"[iter {i}]:{','.join([str(i) for i in new_cents])}\n")
        if np.array_equal(centroids, new_cents):
            break
        centroids = np.copy(new_cents)
        i = i + 1
    distances = scipy.spatial.distance.cdist(x, centroids)
    c_indexes = np.argmin(distances, axis=1) #find the cent index for each point
    new_values = []
    for i in range(len(c_indexes)):
        np.append(new_values, new_cents[c_indexes[i]])
    scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))  # saving


def create_clusters(distances, points):
    num_of_centroid = np.shape(distances)[1]
    num_of_points = np.shape(distances)[0]
    clusters = []  # array of point for each centroid
    closest = np.argmin(distances, axis=1)

    for i in range(num_of_centroid):
        arr = []
        clusters.append(arr)

    for p in range(num_of_points):
        c = closest[p]
        clusters[c].append(points[p])
    return clusters


def create_new_cents(clusters):
    new_cents = []
    for c in clusters:
        new_c = np.mean(c, axis=0)
        new_c = np.round(new_c)
        new_cents.append(new_c)
    return new_cents


if __name__ == '__main__':
    main()
