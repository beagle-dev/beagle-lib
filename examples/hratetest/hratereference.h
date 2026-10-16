#ifndef HRATEREFERENCE_H
#define HRATEREFERENCE_H

#include <cmath>
#include <random>
#include <vector>

static std::vector<double> partialsFromStates(const std::vector<int>& states, int nStates) {
    int n = (int) states.size();
    std::vector<double> p(n * nStates, 0.0);
    for (int i = 0; i < n; ++i) {
        if (states[i] < nStates) {
            p[i * nStates + states[i]] = 1.0;
        } else {
            for (int k = 0; k < nStates; ++k)
                p[i * nStates + k] = 1.0;
        }
    }
    return p;
}

struct StateTree {
    int S;
    std::vector<int> parent;
    std::vector<int> left;
    std::vector<int> right;
    std::vector<double> lambda;
    std::vector<double> pi;
    int root() const { return 2 * S - 2; }
    int nodeCount() const { return 2 * S - 1; }
};

static void finishTree(StateTree& tree, const std::vector<double>& tipPi) {
    int N = tree.nodeCount();
    tree.pi.assign(N, 0.0);
    for (int x = 0; x < tree.S; ++x)
        tree.pi[x] = tipPi[x];
    for (int m = tree.S; m < N; ++m)
        tree.pi[m] = tree.pi[tree.left[m]] + tree.pi[tree.right[m]];
}

static void join(StateTree& tree, int p, int l, int r) {
    tree.left[p] = l;
    tree.right[p] = r;
    tree.parent[l] = p;
    tree.parent[r] = p;
}

static StateTree makeStarTree(int k, const std::vector<double>& tipPi) {
    StateTree tree;
    tree.S = k;
    int N = tree.nodeCount();
    tree.parent.assign(N, -1);
    tree.left.assign(N, -1);
    tree.right.assign(N, -1);
    tree.lambda.assign(N, 0.0);
    join(tree, k, 0, 1);
    for (int i = 1; i < k - 1; ++i)
        join(tree, k + i, k + i - 1, i + 1);
    tree.lambda[tree.root()] = (double) k / (k - 1);
    finishTree(tree, tipPi);
    return tree;
}

static StateTree makeAGCTTree(double lambdaRoot, double lambdaAG, double lambdaCT,
                              const std::vector<double>& tipPi) {
    StateTree tree;
    tree.S = 4;
    int N = tree.nodeCount();
    tree.parent.assign(N, -1);
    tree.left.assign(N, -1);
    tree.right.assign(N, -1);
    tree.lambda.assign(N, 0.0);
    join(tree, 4, 0, 2);
    join(tree, 5, 1, 3);
    join(tree, 6, 4, 5);
    tree.lambda[4] = lambdaAG;
    tree.lambda[5] = lambdaCT;
    tree.lambda[6] = lambdaRoot;
    finishTree(tree, tipPi);
    return tree;
}

static void postOrder(const StateTree& tree, int m, std::vector<int>& out) {
    if (m < tree.S)
        return;
    postOrder(tree, tree.left[m], out);
    postOrder(tree, tree.right[m], out);
    out.push_back(m);
}

static void nodeScalars(const StateTree& tree, double t, std::vector<double>& prZ0, std::vector<double>& H) {
    int N = tree.nodeCount();
    std::vector<int> post;
    postOrder(tree, tree.root(), post);
    prZ0.assign(N, 0.0);
    H.assign(N, 0.0);
    prZ0[tree.root()] = 1.0;
    for (int i = (int) post.size() - 1; i >= 0; --i) {
        int m = post[i];
        double pass = prZ0[m] * std::exp(-tree.lambda[m] * t);
        prZ0[tree.left[m]] = pass;
        prZ0[tree.right[m]] = pass;
    }
    for (int m = 0; m < N; ++m) {
        if (m < tree.S)
            H[m] = prZ0[m] / tree.pi[m];
        else
            H[m] = prZ0[m] * -std::expm1(-tree.lambda[m] * t) / tree.pi[m];
    }
}


static int bundleLength(int S, int categoryCount) {
    int n = S - 1;
    return 1 + 6 * n + categoryCount * (n + S);
}

static void referenceBundle(const StateTree& tree, const double* rates, int categoryCount, double t,
                            std::vector<double>& out, const std::vector<int>* order = nullptr,
                            bool swapChildren = false) {
    int S = tree.S;
    int n = S - 1;
    out.assign(bundleLength(S, categoryCount), 0.0);
    std::vector<int> post;
    if (order)
        post = *order;
    else
        postOrder(tree, tree.root(), post);
    out[0] = 1 * 65536 + n;
    for (int i = 0; i < n; ++i) {
        int m = post[i];
        int a = swapChildren ? tree.right[m] : tree.left[m];
        int b = swapChildren ? tree.left[m] : tree.right[m];
        int j = n - 1 - i;
        out[1 + 3 * i + 0] = m;
        out[1 + 3 * i + 1] = a;
        out[1 + 3 * i + 2] = b;
        out[1 + 3 * n + 3 * j + 0] = m;
        out[1 + 3 * n + 3 * j + 1] = a;
        out[1 + 3 * n + 3 * j + 2] = b;
    }
    std::vector<double> prZ0, H;
    for (int l = 0; l < categoryCount; ++l) {
        nodeScalars(tree, t * rates[l], prZ0, H);
        double* block = out.data() + 1 + 6 * n + l * (n + S);
        for (int m = S; m < 2 * S - 1; ++m)
            block[m - S] = H[m];
        for (int x = 0; x < S; ++x)
            block[n + x] = prZ0[x];
    }
}

static int mrca(const StateTree& tree, int a, int b) {
    std::vector<bool> seen(tree.nodeCount(), false);
    for (int m = a; m >= 0; m = tree.parent[m])
        seen[m] = true;
    for (int m = b; m >= 0; m = tree.parent[m])
        if (seen[m])
            return m;
    return tree.root();
}

static void referenceP(const StateTree& tree, const double* rates, int categoryCount, double t,
                       std::vector<double>& out) {
    int S = tree.S;
    out.assign(categoryCount * S * S, 0.0);
    std::vector<double> prZ0, H, C(tree.nodeCount());
    std::vector<int> post;
    postOrder(tree, tree.root(), post);
    for (int l = 0; l < categoryCount; ++l) {
        nodeScalars(tree, t * rates[l], prZ0, H);
        C[tree.root()] = H[tree.root()];
        for (int i = (int) post.size() - 1; i >= 0; --i) {
            int m = post[i];
            C[tree.left[m]] = H[tree.left[m]] + C[m];
            C[tree.right[m]] = H[tree.right[m]] + C[m];
        }
        for (int x = 0; x < S; ++x)
            for (int y = 0; y < S; ++y)
                out[l * S * S + x * S + y] = tree.pi[y] * C[x == y ? x : mrca(tree, x, y)];
    }
}

template <typename Rng>
static StateTree makeRandomTree(int S, Rng& rng, double lambdaMean) {
    StateTree tree;
    tree.S = S;
    int N = tree.nodeCount();
    tree.parent.assign(N, -1);
    tree.left.assign(N, -1);
    tree.right.assign(N, -1);
    tree.lambda.assign(N, 0.0);
    std::vector<int> pool(S);
    for (int x = 0; x < S; ++x)
        pool[x] = x;
    for (int m = S; m < N; ++m) {
        int i = (int) (rng() % pool.size());
        int a = pool[i];
        pool.erase(pool.begin() + i);
        int j = (int) (rng() % pool.size());
        int b = pool[j];
        pool.erase(pool.begin() + j);
        join(tree, m, a, b);
        pool.push_back(m);
    }
    std::exponential_distribution<double> lambda(1.0 / lambdaMean);
    std::uniform_real_distribution<double> weight(0.2, 1.0);
    for (int m = S; m < N; ++m)
        tree.lambda[m] = lambda(rng);
    std::vector<double> tipPi(S);
    double total = 0.0;
    for (int x = 0; x < S; ++x)
        total += (tipPi[x] = weight(rng));
    for (int x = 0; x < S; ++x)
        tipPi[x] /= total;
    finishTree(tree, tipPi);
    return tree;
}

template <typename Rng>
static std::vector<int> randomPostOrder(const StateTree& tree, Rng& rng) {
    int S = tree.S;
    std::vector<int> ready, order;
    std::vector<int> pending(tree.nodeCount(), 0);
    for (int m = S; m < tree.nodeCount(); ++m) {
        pending[m] = (tree.left[m] >= S) + (tree.right[m] >= S);
        if (pending[m] == 0)
            ready.push_back(m);
    }
    while (!ready.empty()) {
        int i = (int) (rng() % ready.size());
        int m = ready[i];
        ready.erase(ready.begin() + i);
        order.push_back(m);
        int p = tree.parent[m];
        if (p >= 0 && --pending[p] == 0)
            ready.push_back(p);
    }
    return order;
}

static void tn93P(const std::vector<double>& pi, double lambdaRoot, double lambdaAG, double lambdaCT,
                  const double* rates, int categoryCount, double edgeLength, std::vector<double>& out) {
    const int S = 4;
    const double piR = pi[0] + pi[2], piY = pi[1] + pi[3];
    const double beta = lambdaRoot;
    const double alphaR = lambdaRoot + lambdaAG / piR, alphaY = lambdaRoot + lambdaCT / piY;
    const int cls[4] = { 0, 1, 0, 1 };
    const double classPi[2] = { piR, piY };
    const double classAlpha[2] = { alphaR, alphaY };
    out.assign((size_t) categoryCount * S * S, 0.0);
    for (int l = 0; l < categoryCount; ++l) {
        double t = edgeLength * rates[l];
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                double v;
                if (cls[i] != cls[j]) {
                    v = pi[j] * (1.0 - std::exp(-beta * t));
                } else {
                    int c = cls[i];
                    double piC = classPi[c], piO = classPi[1 - c];
                    double rateC = piC * classAlpha[c] + piO * beta;
                    v = pi[j] + pi[j] * (piO / piC) * std::exp(-beta * t) +
                        ((i == j ? 1.0 : 0.0) - pi[j] / piC) * std::exp(-rateC * t);
                }
                out[(size_t) l * S * S + i * S + j] = v;
            }
        }
    }
}

#endif
