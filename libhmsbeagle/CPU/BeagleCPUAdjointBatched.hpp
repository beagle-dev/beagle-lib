/*
 *  BeagleCPUAdjointBatched.hpp
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * PROTOTYPE: the adjoint gradient of the rate matrix in the eigen basis over every branch, pattern and category at
 * once (BEAGLE_ADJOINT_BATCHED=1), for every double-precision CPU implementation (BeagleCPUImpl, used by the
 * generic, SSE and 4-state SSE adjoints). It is used only with few patterns per state (patterns * ratio <= states,
 * BEAGLE_ADJOINT_BATCHED_RATIO, default 6), where the per-branch kernel's own cost dominates.
 *
 * Per column m (a branch, pattern and category) with u = w V^T pre, v = V^{-1} post and time t, the eigen-basis
 * gradient is G = sum_m int_0^t e^{A s} u v^T e^{A (t - s)} ds with A = D^T (D the real block-diagonal eigenvalue
 * matrix). Each term K solves A K - K A = e^{A t} u v^T - u (e^{A^T t} v)^T, so the sum G solves
 *
 *     A G - G A = R,   R = Ue^T X - U^T Xe
 *
 * where the rows of U, X are the u, v of each column and Ue, Xe the same rows times e^{A t}, e^{A^T t}. U and X
 * are two GEMMs (the rotations), R two more; the Sylvester equation is solved block by block (1x1 real, 2x2
 * complex pair). Blocks whose eigenvalues (nearly) coincide make it singular; those entries are summed directly
 * over the columns: clusters of close real eigenvalues by a Taylor series around the cluster mean (a few small
 * GEMMs), and the diagonal block of each complex pair in closed form. Other degenerate blocks make the function
 * return false, and the caller uses the per-branch kernel.
 *
 * @author Marc Suchard
 */

#ifndef BEAGLE_CPU_ADJOINT_BATCHED_HPP
#define BEAGLE_CPU_ADJOINT_BATCHED_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(__APPLE__) && defined(BEAGLE_HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace beagle {
namespace cpu {

namespace adjoint_batched {

// Row-major C (m x n) = alpha op(A) B + beta C, op(A) = A (m x k, row stride lda) or A^T (A stored k x m)
inline void gemm(bool transA, int m, int n, int k, double alpha, const double* A, int lda,
                 const double* B, int ldb, double beta, double* C, int ldc) {
#if defined(__APPLE__) && defined(BEAGLE_HAVE_ACCELERATE)
    cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    for (int i = 0; i < m; ++i) {
        double* c = C + i * ldc;
        for (int j = 0; j < n; ++j) {
            c[j] = (beta == 0.0) ? 0.0 : beta * c[j];
        }
        for (int p = 0; p < k; ++p) {
            const double a = alpha * (transA ? A[p * lda + i] : A[i * lda + p]);
            if (a != 0.0) {
                const double* b = B + p * ldb;
                for (int j = 0; j < n; ++j) {
                    c[j] += a * b[j];
                }
            }
        }
    }
#endif
}

// A real eigenvalue a (size 1), or a conjugate pair a +/- i b at (start, start + 1) (size 2)
struct Block {
    int start;
    int size;
    double a;
    double b;
};

// The block of A = D^T: [a] or [[a, -b], [b, a]] (D's block is [[a, b], [-b, a]], so e^{D t} rotates by +bt)
inline void blockMatrix(const Block& block, double* out) {
    if (block.size == 1) {
        out[0] = block.a;
    } else {
        out[0] = block.a;  out[1] = -block.b;
        out[2] = block.b;  out[3] = block.a;
    }
}

inline double spectrumDistance(const Block& I, const Block& J) {
    double distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < I.size; ++i) {
        const std::complex<double> mu(I.a, (I.size == 1) ? 0.0 : (i == 0 ? I.b : -I.b));
        for (int j = 0; j < J.size; ++j) {
            const std::complex<double> nu(J.a, (J.size == 1) ? 0.0 : (j == 0 ? J.b : -J.b));
            distance = std::min(distance, std::abs(mu - nu));
        }
    }
    return distance;
}

// X (p x q, row-major) with AI X - X AJ = R; Gaussian elimination with partial pivoting on the pq x pq system
inline bool solveSylvesterBlock(const double* AI, int p, const double* AJ, int q, const double* R, double* X) {
    const int n = p * q;
    double K[16];
    double y[4];
    for (int e = 0; e < n * n; ++e) {
        K[e] = 0.0;
    }
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < q; ++j) {
            const int row = i * q + j;
            y[row] = R[row];
            for (int k = 0; k < p; ++k) {
                K[row * n + (k * q + j)] += AI[i * p + k];
            }
            for (int k = 0; k < q; ++k) {
                K[row * n + (i * q + k)] -= AJ[k * q + j];
            }
        }
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int r = col + 1; r < n; ++r) {
            if (std::abs(K[r * n + col]) > std::abs(K[pivot * n + col])) {
                pivot = r;
            }
        }
        if (K[pivot * n + col] == 0.0) {
            return false;
        }
        if (pivot != col) {
            for (int c = 0; c < n; ++c) {
                std::swap(K[col * n + c], K[pivot * n + c]);
            }
            std::swap(y[col], y[pivot]);
        }
        for (int r = col + 1; r < n; ++r) {
            const double factor = K[r * n + col] / K[col * n + col];
            for (int c = col; c < n; ++c) {
                K[r * n + c] -= factor * K[col * n + c];
            }
            y[r] -= factor * y[col];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        double sum = y[r];
        for (int c = r + 1; c < n; ++c) {
            sum -= K[r * n + c] * X[c];
        }
        X[r] = sum / K[r * n + r];
    }
    return true;
}

// y = e^{D t} x (Backward = false) or e^{D^T t} x (Backward = true), as spectral_sse::expScale
template <bool Backward>
inline void expRotate(double* y, const double* x, const double* expat, const double* expatcosbt,
                      const double* expatsinbt, const double* imag, int S) {
    for (int i = 0; i < S; ) {
        if (imag == nullptr || imag[i] == 0.0) {
            y[i] = expat[i] * x[i];
            ++i;
        } else {
            const double c = expatcosbt[i];
            const double s = Backward ? -expatsinbt[i] : expatsinbt[i];
            const double a = x[i];
            const double b = x[i + 1];
            y[i]     = c * a + s * b;
            y[i + 1] = c * b - s * a;
            i += 2;
        }
    }
}

} // namespace adjoint_batched

BEAGLE_CPU_TEMPLATE
bool BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcAdjointCrossProductsBatched(const int* postBufferIndices,
                                                                   const int* preBufferIndices,
                                                                   const int* branchEigenIndices,
                                                                   const double* categoryRates,
                                                                   const REALTYPE* categoryWeights,
                                                                   const REALTYPE* perSiteLikelihoods,
                                                                   REALTYPE* buffer,
                                                                   int startNode,
                                                                   int endNode,
                                                                   int startPattern,
                                                                   int endPattern,
                                                                   int currentPartition,
                                                                   const int* postScaleIndices,
                                                                   const int* preScaleIndices,
                                                                   const int cumulativeScaleIndex) {
    if constexpr (!std::is_same_v<REALTYPE, double>) {
        return false; // double precision only
    } else {
    using namespace adjoint_batched;
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const int S = kStateCount;
    const int stride = kTransPaddedStateCount;
    if (endNode <= startNode) {
        return true;
    }

    const int eigenIndex = gBranchEigenInfo[branchEigenIndices[startNode]].eigenIndex;
    for (int node = startNode; node < endNode; ++node) {
        if (gBranchEigenInfo[branchEigenIndices[node]].eigenIndex != eigenIndex) {
            return false; // TODO one batch per eigen decomposition
        }
    }

    const double* eval = gEigenDecomposition->getEigenValuesPtr(eigenIndex);
    const double* imag = (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) ? eval + S : nullptr;

    std::vector<Block> blocks;
    for (int i = 0; i < S; ) {
        if (imag == nullptr || imag[i] == 0.0) {
            blocks.push_back(Block{i, 1, eval[i], 0.0});
            ++i;
        } else {
            blocks.push_back(Block{i, 2, eval[i], imag[i]});
            i += 2;
        }
    }

    // Columns: category, node, pattern with non-zero weight
    int patternsUsed = 0;
    for (int k = startPattern; k < endPattern; ++k) {
        if (gPatternWeights[k] != 0.0) {
            ++patternsUsed;
        }
    }
    if (patternsUsed * kBatchedAdjointRatio > S) {
        return false; // the per-branch kernel is faster
    }
    const int M = kCategoryCount * (endNode - startNode) * patternsUsed;
    if (M == 0) {
        return true;
    }

    if (static_cast<int>(gBatchedPanels.size()) < kPartitionCount) {
        gBatchedPanels.resize(kPartitionCount);
    }
    std::vector<double>& panels = gBatchedPanels[currentPartition];
    if (panels.size() < static_cast<size_t>(4) * M * S) {
        panels.resize(static_cast<size_t>(4) * M * S);
    }
    double* pre = panels.data();          // w * pre partial; later Ue = e^{A t} u
    double* post = pre + size_t(M) * S;   // post partial;      later Xe = e^{A^T t} v
    double* U = post + size_t(M) * S;     // u = w V^T pre
    double* X = U + size_t(M) * S;        // v = V^{-1} post

    std::vector<double> columnTime(M);
    std::vector<const BranchEigenInfo*> columnInfo(M);
    std::vector<int> columnOffset(M);

    int m = 0;
    double maxTime = 0.0;
    for (int category = 0; category < kCategoryCount; ++category) {
        const double categoryRate = categoryRates[category];
        const double categoryWeight = categoryWeights[category];
        const int infoOffset = category * kPartialsPaddedStateCount;

        for (int node = startNode; node < endNode; ++node) {
            const BranchEigenInfo& info = gBranchEigenInfo[branchEigenIndices[node]];
            const double time = categoryRate * info.branchLength;
            maxTime = std::max(maxTime, time);

            const double* prePartials = gPartials[preBufferIndices[node]];
            const int* tipStates = gTipStates[postBufferIndices[node]];
            const double* postPartials = (tipStates != nullptr) ? nullptr : gPartials[postBufferIndices[node]];

            const double* nodeLikelihoods = adjointNodeLikelihoods(node, perSiteLikelihoods, postScaleIndices,
                    preScaleIndices, cumulativeScaleIndex, startPattern, endPattern, currentPartition);

            for (int k = startPattern; k < endPattern; ++k) {
                if (gPatternWeights[k] == 0.0) {
                    continue;
                }
                const int v = category * kPartialsPaddedStateCount * kPatternCount + kPartialsPaddedStateCount * k;
                const double weight = gPatternWeights[k] * categoryWeight / nodeLikelihoods[k];

                double* preRow = pre + size_t(m) * S;
                for (int i = 0; i < S; ++i) {
                    preRow[i] = weight * prePartials[v + i];
                }
                double* postRow = post + size_t(m) * S;
                if (postPartials != nullptr) {
                    std::copy(postPartials + v, postPartials + v + S, postRow);
                } else if (tipStates[k] < S) {
                    std::fill(postRow, postRow + S, 0.0);
                    postRow[tipStates[k]] = 1.0;
                } else { // missing state
                    std::fill(postRow, postRow + S, 1.0);
                }

                columnTime[m] = time;
                columnInfo[m] = &info;
                columnOffset[m] = infoOffset;
                ++m;
            }
        }
    }

    const auto t1 = Clock::now();

    // Rotations: U = pre V (rows u = V^T pre), X = post (V^{-1})^T (rows v = V^{-1} post)
    gemm(false, M, S, S, 1.0, pre, S, gEigenDecomposition->getEigenVectorsPtr(eigenIndex), stride, 0.0, U, S);
    gemm(false, M, S, S, 1.0, post, S, gEigenDecomposition->getBackwardsEigenVectorsPtr(eigenIndex), stride,
         0.0, X, S);

    const auto t2 = Clock::now();

    // Ue = e^{A t} u = e^{D^T t} u, Xe = e^{A^T t} v = e^{D t} v, over the input panels
    for (int c = 0; c < M; ++c) {
        const BranchEigenInfo& info = *columnInfo[c];
        const int offset = columnOffset[c];
        expRotate<true>(pre + size_t(c) * S, U + size_t(c) * S, info.expat + offset,
                        info.expatcosbt + offset, info.expatsinbt + offset, imag, S);
        expRotate<false>(post + size_t(c) * S, X + size_t(c) * S, info.expat + offset,
                         info.expatcosbt + offset, info.expatsinbt + offset, imag, S);
    }

    // R = Ue^T X - U^T Xe
    std::vector<double> R(size_t(S) * S);
    gemm(true, S, S, M, 1.0, pre, S, X, S, 0.0, R.data(), S);
    gemm(true, S, S, M, -1.0, U, S, post, S, 1.0, R.data(), S);

    const auto t3 = Clock::now();

    // Block-wise Sylvester solve, with degenerate blocks summed directly
    std::vector<double> G(size_t(S) * S, 0.0);
    int degenerateReal = 0;
    int degenerateEqual = 0;
    int degeneratePair = 0;

    // Clusters of real eigenvalues: single linkage on |a_l - a_r| * maxTime < delta. Every pair inside a cluster,
    // close or not, is summed by the series below; pairs from different clusters are never degenerate.
    std::vector<int> clusterOf(S, -1);
    std::vector<std::vector<int>> clusters;
    {
        std::vector<int> reals;
        for (const Block& I : blocks) {
            if (I.size == 1) {
                reals.push_back(I.start);
            }
        }
        std::sort(reals.begin(), reals.end(), [eval](int x, int y) { return eval[x] < eval[y]; });
        for (size_t i = 0; i < reals.size(); ++i) {
            if (i == 0 || (eval[reals[i]] - eval[reals[i - 1]]) * maxTime >= kBatchedAdjointDelta) {
                clusters.emplace_back();
            }
            clusters.back().push_back(reals[i]);
            clusterOf[reals[i]] = static_cast<int>(clusters.size()) - 1;
        }
    }

    for (const Block& I : blocks) {
        double AI[4];
        blockMatrix(I, AI);
        for (const Block& J : blocks) {
            if (I.size == 1 && J.size == 1 && clusterOf[I.start] == clusterOf[J.start]) {
                continue; // the cluster series below
            }
            const double distance = spectrumDistance(I, J);
            const bool degenerate = (distance * maxTime < kBatchedAdjointDelta);

            if (!degenerate) {
                double AJ[4], RIJ[4], XIJ[4];
                blockMatrix(J, AJ);
                for (int i = 0; i < I.size; ++i) {
                    for (int j = 0; j < J.size; ++j) {
                        RIJ[i * J.size + j] = R[(I.start + i) * S + J.start + j];
                    }
                }
                if (!solveSylvesterBlock(AI, I.size, AJ, J.size, RIJ, XIJ)) {
                    return false;
                }
                for (int i = 0; i < I.size; ++i) {
                    for (int j = 0; j < J.size; ++j) {
                        G[(I.start + i) * S + J.start + j] = XIJ[i * J.size + j];
                    }
                }
            } else if (I.start == J.start) {
                // Diagonal block of a conjugate pair: int_0^t e^{A s} M e^{A (t - s)} ds with e^{A s} =
                // e^{a s} (cos(bs) I - sin(bs) J), J = [[0, 1], [-1, 0]]:
                // e^{a t} (alpha M - beta (M J + J M) + gamma J M J)
                ++degeneratePair;
                const int i0 = I.start;
                const double b = I.b;
                double K00 = 0.0, K01 = 0.0, K10 = 0.0, K11 = 0.0;
                for (int c = 0; c < M; ++c) {
                    const double t = columnTime[c];
                    const BranchEigenInfo& info = *columnInfo[c];
                    const int offset = columnOffset[c];
                    const double e = info.expat[offset + i0];
                    const double cs = info.cosbt[offset + i0];
                    const double sn = info.sinbt[offset + i0];
                    const double alpha = 0.5 * (t * cs + sn / b);
                    const double beta = 0.5 * t * sn;
                    const double gamma = 0.5 * (sn / b - t * cs);

                    const double* u = U + size_t(c) * S + i0;
                    const double* x = X + size_t(c) * S + i0;
                    const double M00 = u[0] * x[0], M01 = u[0] * x[1];
                    const double M10 = u[1] * x[0], M11 = u[1] * x[1];
                    // M J + J M and J M J
                    const double S00 = -M01 + M10, S01 = M00 + M11;
                    const double S10 = -M11 - M00, S11 = M10 - M01;
                    const double T00 = -M11, T01 = M10, T10 = M01, T11 = -M00;

                    K00 += e * (alpha * M00 - beta * S00 + gamma * T00);
                    K01 += e * (alpha * M01 - beta * S01 + gamma * T01);
                    K10 += e * (alpha * M10 - beta * S10 + gamma * T10);
                    K11 += e * (alpha * M11 - beta * S11 + gamma * T11);
                }
                G[i0 * S + i0] = K00;
                G[i0 * S + i0 + 1] = K01;
                G[(i0 + 1) * S + i0] = K10;
                G[(i0 + 1) * S + i0 + 1] = K11;
            } else {
                if (kBatchedAdjointVerbose) {
                    fprintf(stderr, "batched adjoint: degenerate block (%d, %d) sizes (%d, %d), distance %g; "
                                    "using the per-branch kernel\n", I.start, J.start, I.size, J.size, distance);
                }
                return false;
            }
        }
    }

    // Real clusters C: int_0^t e^{a_l s} e^{a_r (t - s)} ds = e^{abar t} sum_k t^{k+1} / (k+1)! h_k(d_l, d_r) with
    // d = a - abar and h_k(x, y) = sum_{p+q=k} x^p y^q, so G_CC = sum_k h_k o (Uc^T diag(w_k) Xc) with
    // w_k = e^{abar t} t^{k+1} / (k+1)!, truncated where (max |d| maxTime)^{K+1} / (K+1)! < 1e-17
    {
        std::vector<double> Uc, Xc, W, H, eBar(M), power(M);
        for (const auto& cluster : clusters) {
            const int c = static_cast<int>(cluster.size());
            degenerateReal += c * c;
            for (int l : cluster) {
                for (int r : cluster) {
                    if (eval[l] == eval[r]) {
                        ++degenerateEqual;
                    }
                }
            }

            if (c == 1) { // t e^{a t}
                const int l = cluster[0];
                double sum = 0.0;
                for (int m = 0; m < M; ++m) {
                    sum += U[size_t(m) * S + l] * X[size_t(m) * S + l] *
                           columnTime[m] * columnInfo[m]->expat[columnOffset[m] + l];
                }
                G[l * S + l] = sum;
                continue;
            }

            double mean = 0.0;
            for (int l : cluster) {
                mean += eval[l];
            }
            mean /= c;
            std::vector<double> d(c);
            double maxAbs = 0.0;
            for (int i = 0; i < c; ++i) {
                d[i] = eval[cluster[i]] - mean;
                maxAbs = std::max(maxAbs, std::abs(d[i]));
            }
            const double x = maxAbs * maxTime;
            if (x > 0.5) {
                if (kBatchedAdjointVerbose) {
                    fprintf(stderr, "batched adjoint: real cluster of %d spans %g * maxTime; using the per-branch "
                                    "kernel\n", c, 2 * maxAbs);
                }
                return false;
            }
            int K = 0;
            for (double term = x; term > 1e-17 && K < 30; ) {
                ++K;
                term *= x / (K + 1);
            }

            Uc.resize(size_t(M) * c);
            Xc.resize(size_t(M) * c);
            W.resize(size_t(M) * c);
            H.resize(size_t(c) * c);
            for (int m = 0; m < M; ++m) {
                for (int i = 0; i < c; ++i) {
                    Uc[size_t(m) * c + i] = U[size_t(m) * S + cluster[i]];
                    Xc[size_t(m) * c + i] = X[size_t(m) * S + cluster[i]];
                }
                eBar[m] = std::exp(mean * columnTime[m]);
                power[m] = columnTime[m]; // t^{k+1} / (k+1)!
            }
            for (int k = 0; k <= K; ++k) {
                for (int m = 0; m < M; ++m) {
                    const double w = eBar[m] * power[m];
                    for (int i = 0; i < c; ++i) {
                        W[size_t(m) * c + i] = w * Uc[size_t(m) * c + i];
                    }
                    power[m] *= columnTime[m] / (k + 2);
                }
                gemm(true, c, c, M, 1.0, W.data(), c, Xc.data(), c, 0.0, H.data(), c);
                for (int i = 0; i < c; ++i) {
                    for (int j = 0; j < c; ++j) {
                        double h = 0.0; // sum_{p=0}^{k} d_i^p d_j^{k-p}
                        double di = 1.0;
                        for (int p = 0; p <= k; ++p) {
                            h += di * std::pow(d[j], k - p);
                            di *= d[i];
                        }
                        G[cluster[i] * S + cluster[j]] += h * H[i * c + j];
                    }
                }
            }
        }
    }

    const auto t4 = Clock::now();

    if (kBatchedAdjointCheck) {
        std::vector<double> reference(size_t(S) * S, 0.0);
        batchedAdjointCheckActive() = true;
        calcAdjointCrossProductsRange(postBufferIndices, preBufferIndices, branchEigenIndices, categoryRates,
                                      categoryWeights, perSiteLikelihoods, reference.data(), startNode, endNode,
                                      startPattern, endPattern, currentPartition, postScaleIndices,
                                      preScaleIndices, cumulativeScaleIndex, nullptr);
        batchedAdjointCheckActive() = false;
        double maxReference = 0.0, maxDifference = 0.0;
        int worst = 0;
        for (int e = 0; e < S * S; ++e) {
            maxReference = std::max(maxReference, std::abs(reference[e]));
            const double difference = std::abs(reference[e] - G[e]);
            if (difference > maxDifference) {
                maxDifference = difference;
                worst = e;
            }
        }
        const int wl = worst / S, wr = worst % S;
        fprintf(stderr, "batched adjoint check: max |ref| %.6e, max |diff| %.3e (relative %.3e) at (%d, %d): "
                        "ref %.12e batched %.12e; eigen (%.9g%+.9gi, %.9g%+.9gi), |difference| * maxTime %.3e\n",
                maxReference, maxDifference, maxDifference / maxReference, wl, wr, reference[worst], G[worst],
                eval[wl], imag ? imag[wl] : 0.0, eval[wr], imag ? imag[wr] : 0.0,
                std::abs(std::complex<double>(eval[wl], imag ? imag[wl] : 0.0) -
                         std::complex<double>(eval[wr], imag ? imag[wr] : 0.0)) * maxTime);
    }

    for (int e = 0; e < S * S; ++e) {
        buffer[e] += G[e];
    }

    if (kBatchedAdjointVerbose) {
        const auto t5 = Clock::now();
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr, "batched adjoint: M %d, blocks %d, maxTime %.4g, degenerate real %d (equal %d) pair %d; "
                        "gather %.2f ms, rotate %.2f ms, scale+R %.2f ms, solve %.2f ms, total %.2f ms\n",
                M, static_cast<int>(blocks.size()), maxTime, degenerateReal, degenerateEqual, degeneratePair,
                ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t0, t5));
    }

    return true;
    }
}

} // namespace cpu
} // namespace beagle

#endif // BEAGLE_CPU_ADJOINT_BATCHED_HPP
