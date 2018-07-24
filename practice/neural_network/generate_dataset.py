#!/usr/bin/python

import os
import random


# sample: the source sample to be splited
# percentage: (train, test, verify)
#     default, train %80, test %10, verify %10
# shuffle: shuffle the splitted sample if true
def split_sample(sample, percentage=(80, 10, 10), shuffle=False):

    if shuffle:
        random.shuffle(sample)

    nr_sample = len(sample)

    train_start = 0
    test_start = nr_sample * sum(percentage[0:1]) / sum(percentage)
    verify_start = nr_sample * sum(percentage[0:2]) / sum(percentage)

    train_sample = sample[train_start:test_start]
    test_sample = sample[test_start:verify_start]
    verify_sample = sample[verify_start:]

    print('split_sample: train=%d, test=%d, verify=%d, total=%d'
          % (len(train_sample), len(test_sample), len(verify_sample), nr_sample))

    return train_sample, test_sample, verify_sample


def save_sample(samples, filename):
    header = ",".join(["col{}".format(i) for i in range(len(samples[0].split(",")))])
    with open(filename, 'w') as f:
        f.write(header + "\n")
        for sample in samples:
            f.write(sample)
        print('save %d done' % len(samples))


# y = x1 - x2 + 1
def gen_sample():
    samples = []
    for x1 in range(0, 64):
        for x2 in range(0, 64):
            y = x1 + x2 + 1
            if y < 32:
                sample = '%d,%d,0.1,0.9' % (x1, x2)
            else:
                sample = '%d,%d,0.9,0.1' % (x1, x2)
            samples.append(sample + '\n')
    return samples


def main():
    samples = gen_sample()
    train, test, verify = split_sample(samples, percentage=(50, 25, 25))
    save_sample(train, 'train.csv')
    save_sample(test, 'test.csv')
    save_sample(verify, 'verify.csv')

if __name__ == '__main__':
    main()
