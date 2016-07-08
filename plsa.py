#!/usr/bin/python
# coding: UTF-8

import argparse
import numpy as np
import co_occurrence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str)
    parser.add_argument("test_file", type=str)
    parser.add_argument("--iteration", dest="iteration", type=int, default=1000,
                        help="The maximum number of training iteration")
    parser.add_argument("--threshold", dest="threshold", type=float, default=1.0e-4,
                        help="The threshold of the log likelihood")
    parser.add_argument("--initial", dest="initial", type=int, default=2,
                        help="The initial value of latent variable")
    return parser.parse_args()


class PLSA:
    def __init__(self, N, Z):
        self.N = N
        self.X = N.shape[0]
        self.Y = N.shape[1]
        self.Z = Z

        self.P_z = np.random.rand(self.Z)
        self.P_xz = np.random.rand(self.Z, self.X)
        self.P_yz = np.random.rand(self.Z, self.Y)
        self.P_zxy = np.random.rand(self.X, self.Y, self.Z)

        # normalize
        self.P_z /= np.sum(self.P_z)
        self.P_xz /= np.sum(self.P_xz, axis=1)[:, None]
        self.P_yz /= np.sum(self.P_yz, axis=1)[:, None]

    def e_step(self):
        self.P_zxy = self.P_z[None, None, :] * self.P_xz.T[:, None, :] * self.P_yz.T[None, :, :]
        # deal with nan , inf
        self.P_zxy /= np.sum(self.P_zxy, axis=2)[:, :, None]
        self.P_zxy[np.isinf(self.P_zxy)] = 0
        self.P_zxy[np.isnan(self.P_zxy)] = 0

    def m_step(self):
        NP = self.N[:,:,None] * self.P_zxy# deal with nan , inf
        # deal with nan, inf
        NP[np.isinf(NP)] = 0
        NP[np.isnan(NP)] = 0
        self.P_z = np.sum(NP, axis=(0, 1))
        self.P_xz = np.sum(NP, axis=1).T
        self.P_yz = np.sum(NP, axis=0).T

        # normalization
        self.P_z /= np.sum(self.P_z)
        self.P_xz /= np.sum(self.P_xz, axis=1)[:, None]
        self.P_yz /= np.sum(self.P_yz, axis=1)[:, None]

    def loglikelihood(self):
        P_xy = self.P_z[None, None, :] * self.P_xz.T[:, None, :] * self.P_yz.T[None, :, :]
        # deal with nan, inf
        P_xy[np.isinf(P_xy)] = -1000
        P_xy[np.isnan(P_xy)] = 0
        P_xy = np.sum(P_xy, axis=2)
        # normalization
        P_xy /= np.sum(P_xy)
        logP_xy = np.log(P_xy)
        logP_xy[np.isinf(logP_xy)] = -1000
        return np.sum(self.N * logP_xy)

    def train(self, args):
        prev = 100000
        for i in range(args.iteration):
            self.e_step()
            self.m_step()
            temp = self.loglikelihood()
            pp = self.perplexity()
            print(pp)

            if abs(temp - prev) < args.threshold:
                break
            prev = temp
        print(i)

    def perplexity(self):
        P_xy = self.P_z[None, None, :] * self.P_xz.T[:, None, :] * self.P_yz.T[None, :, :]
        # deal with nan, inf
        P_xy[np.isinf(P_xy)] = 0
        P_xy[np.isnan(P_xy)] = 0
        P_xy = np.sum(P_xy, axis=2)

        logP_xy = np.log2(P_xy) * self.N
        logP_xy[np.isinf(logP_xy)] = 0
        logP_xy[np.isnan(logP_xy)] = 0

        exponent = np.sum(logP_xy, axis=(0,1)) / np.sum(self.N, axis=(0,1))
        return 2 ** (-exponent)


def main():
    args = parse_args()
    # train
    v = co_occurrence.Cooccurence()
    v.make_dictionary(args.train_file)
    N = v.count_occurrence(args.train_file)
    train_plsa = PLSA(N, args.initial)
    train_plsa.train(args)

    # test
    d = co_occurrence.Cooccurence()
    d.make_dictionary(args.train_file)
    N_test = d.count_occurrence(args.test_file)
    test_plsa = PLSA(N_test, args.initial)
    pp = test_plsa.perplexity()
    print(pp)


if __name__ == '__main__':
    main()