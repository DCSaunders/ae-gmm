from __future__ import division
import argparse
import collections
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import cPickle
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', help='File of vectors to fit')
    parser.add_argument('--load_proj', 
                        help='Location to load pickled input data and projected data from')
    parser.add_argument('--load_model',  
                        help='Location to load pickled model from')
    parser.add_argument('--load_lengths', default=False, action='store_true',
                        help='True if loading data of format {states:[] length:x}')
    parser.add_argument('--save_input',
                        help='Location to save pickled input data')
    parser.add_argument('--save_model',
                        help='Location to save pickled model')
    parser.add_argument('--save_samples',
                        help='Location to save pickled samples from model')
    parser.add_argument('--save_labels',
                        help='Location to save labelled training samples')
    parser.add_argument('--pickled_in', action='store_true', default=False,
                        help='If true, input vectors must be unpickled')
    parser.add_argument('--max_in', type=int, default=40000, help='Max number of training inputs - uses all if 0')
    parser.add_argument('--samples_per_mode', type=int, default=100, help='Average number of training samples per mode')
    parser.add_argument('--n_clusters', type=int, default=0,
                        help='Number of clusters. Use rule-of-thumb if 0.')

    parser.add_argument('--interpolate', default='/tmp/interpolate', 
        help='Location to store samples interpolated between training examples')
    parser.add_argument('--uneven', action='store_true', default=False,
                        help='Set if interpolating with uneven intervals')

    return parser.parse_args()


def get_samples(args):
    lengths = None
    if args.pickled_in:
        samples = []
        lengths = []
        with open(args.file_in, 'rb') as f:
            unpickler = cPickle.Unpickler(f)
            while True:
                try:
                    saved = unpickler.load()
                    if args.load_lengths:
                        sample = np.array(saved['states'][0][0], dtype=float)
                        samples.append(sample)
                        lengths.append(saved['length'])
                    else:
                        samples.append(np.array(saved)[0])
                    if args.max_in and len(samples) >= args.max_in:
                        break
                except (EOFError):
                    break
            print 'Unpickled {} samples from {}'.format(
                len(samples), args.file_in)
    else:
        samples = np.loadtxt(args.file_in)
        print 'Loaded samples file from {}'.format(args.file_in)
    return samples, lengths

def do_interpolate(train_labels_samples, interpolate_count, interp_file, uneven):
    label_interps_dict = collections.defaultdict(list)
    sampled_labels = collections.defaultdict(list)
    for entry in train_labels_samples:
        sampled_labels[entry[0]].append(entry[1])
    for k, v in sampled_labels.items():
        print k, len(v)
    count = 0
    while count < interpolate_count:
        for label in sampled_labels:
            if len(sampled_labels[label]) >= 2:
                interpolations = [sampled_labels[label].pop()]
                final_sample = sampled_labels[label].pop()
                if uneven:
                    uneven_interpolation(interpolations, final_sample)
                else:
                    even_interpolation(interpolations, final_sample, interpolate_count)
                label_interps_dict[label].append(interpolations)
                count += 1
    with open(interp_file, 'wb') as f:
        cPickle.dump(label_interps_dict, f)


def uneven_interpolation(interpolations, endpoint):
    diff = endpoint - interpolations[0]
    for incr in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        interpolations.append(interpolations[0] + incr * diff)


def even_interpolation(interpolations, endpoint):
    increment = (endpoint - interpolations[0]) / interpolate_count 
    for interp_num in range(0, interpolate_count):
        interpolations.append(interpolations[interp_num] + increment)


def cluster(samples, n_clusters=0, samples_per_mode=100):
    if n_clusters == 0:
        # rule-of-thumb fitting
        n_clusters = int(len(samples) / samples_per_mode)
    kmeans = KMeans(n_clusters=n_clusters).fit(samples)
    return zip(kmeans.labels_, samples)
    
if __name__ == '__main__':
    args = get_args()
    np.random.seed(1234)
    n_interpolate = 6
    samples, lengths = get_samples(args)
    labelled_samples = cluster(samples, args.n_clusters, args.samples_per_mode)
    do_interpolate(labelled_samples, n_interpolate, args.interpolate, args.uneven)
      
