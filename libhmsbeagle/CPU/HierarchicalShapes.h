/*
 *  HierarchicalShapes.h
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#ifndef HIERARCHICALSHAPES_H_
#define HIERARCHICALSHAPES_H_

#include <cstring>

namespace beagle {
namespace cpu {

const int kHierarchicalShapeTableMaxStates = 8;

template <int S, int ID>
struct HierarchicalShape;

template <int S>
struct HierarchicalShapeCount {
    static constexpr int value = 0;
};

template <>
struct HierarchicalShapeCount<2> {
    static constexpr int value = 1;
};

template <>
struct HierarchicalShape<2, 0> {
    static constexpr int post[3] = { 2, 0, 1 };
    static constexpr int pre[3] = { 2, 0, 1 };
};

template <>
struct HierarchicalShapeCount<3> {
    static constexpr int value = 1;
};

template <>
struct HierarchicalShape<3, 0> {
    static constexpr int post[6] = { 3, 0, 1, 4, 3, 2 };
    static constexpr int pre[6] = { 4, 3, 2, 3, 0, 1 };
};

template <>
struct HierarchicalShapeCount<4> {
    static constexpr int value = 2;
};

template <>
struct HierarchicalShape<4, 0> {
    static constexpr int post[9] = { 4, 0, 1, 5, 4, 2, 6, 5, 3 };
    static constexpr int pre[9] = { 6, 5, 3, 5, 4, 2, 4, 0, 1 };
};

template <>
struct HierarchicalShape<4, 1> {
    static constexpr int post[9] = { 4, 0, 1, 5, 2, 3, 6, 4, 5 };
    static constexpr int pre[9] = { 6, 4, 5, 4, 0, 1, 5, 2, 3 };
};

template <>
struct HierarchicalShapeCount<5> {
    static constexpr int value = 3;
};

template <>
struct HierarchicalShape<5, 0> {
    static constexpr int post[12] = { 5, 0, 1, 6, 5, 2, 7, 6, 3, 8, 7, 4 };
    static constexpr int pre[12] = { 8, 7, 4, 7, 6, 3, 6, 5, 2, 5, 0, 1 };
};

template <>
struct HierarchicalShape<5, 1> {
    static constexpr int post[12] = { 5, 0, 1, 6, 2, 3, 7, 5, 6, 8, 7, 4 };
    static constexpr int pre[12] = { 8, 7, 4, 7, 5, 6, 5, 0, 1, 6, 2, 3 };
};

template <>
struct HierarchicalShape<5, 2> {
    static constexpr int post[12] = { 5, 0, 1, 6, 5, 2, 7, 3, 4, 8, 6, 7 };
    static constexpr int pre[12] = { 8, 6, 7, 6, 5, 2, 5, 0, 1, 7, 3, 4 };
};

template <>
struct HierarchicalShapeCount<6> {
    static constexpr int value = 6;
};

template <>
struct HierarchicalShape<6, 0> {
    static constexpr int post[15] = { 6, 0, 1, 7, 6, 2, 8, 7, 3, 9, 8, 4, 10, 9, 5 };
    static constexpr int pre[15] = { 10, 9, 5, 9, 8, 4, 8, 7, 3, 7, 6, 2, 6, 0, 1 };
};

template <>
struct HierarchicalShape<6, 1> {
    static constexpr int post[15] = { 6, 0, 1, 7, 2, 3, 8, 6, 7, 9, 8, 4, 10, 9, 5 };
    static constexpr int pre[15] = { 10, 9, 5, 9, 8, 4, 8, 6, 7, 6, 0, 1, 7, 2, 3 };
};

template <>
struct HierarchicalShape<6, 2> {
    static constexpr int post[15] = { 6, 0, 1, 7, 6, 2, 8, 3, 4, 9, 7, 8, 10, 9, 5 };
    static constexpr int pre[15] = { 10, 9, 5, 9, 7, 8, 7, 6, 2, 6, 0, 1, 8, 3, 4 };
};

template <>
struct HierarchicalShape<6, 3> {
    static constexpr int post[15] = { 6, 0, 1, 7, 6, 2, 8, 7, 3, 9, 4, 5, 10, 8, 9 };
    static constexpr int pre[15] = { 10, 8, 9, 8, 7, 3, 7, 6, 2, 6, 0, 1, 9, 4, 5 };
};

template <>
struct HierarchicalShape<6, 4> {
    static constexpr int post[15] = { 6, 0, 1, 7, 2, 3, 8, 6, 7, 9, 4, 5, 10, 8, 9 };
    static constexpr int pre[15] = { 10, 8, 9, 8, 6, 7, 6, 0, 1, 7, 2, 3, 9, 4, 5 };
};

template <>
struct HierarchicalShape<6, 5> {
    static constexpr int post[15] = { 6, 0, 1, 7, 6, 2, 8, 3, 4, 9, 8, 5, 10, 7, 9 };
    static constexpr int pre[15] = { 10, 7, 9, 7, 6, 2, 6, 0, 1, 9, 8, 5, 8, 3, 4 };
};

template <>
struct HierarchicalShapeCount<7> {
    static constexpr int value = 11;
};

template <>
struct HierarchicalShape<7, 0> {
    static constexpr int post[18] = { 7, 0, 1, 8, 7, 2, 9, 8, 3, 10, 9, 4, 11, 10, 5, 12, 11, 6 };
    static constexpr int pre[18] = { 12, 11, 6, 11, 10, 5, 10, 9, 4, 9, 8, 3, 8, 7, 2, 7, 0, 1 };
};

template <>
struct HierarchicalShape<7, 1> {
    static constexpr int post[18] = { 7, 0, 1, 8, 2, 3, 9, 7, 8, 10, 9, 4, 11, 10, 5, 12, 11, 6 };
    static constexpr int pre[18] = { 12, 11, 6, 11, 10, 5, 10, 9, 4, 9, 7, 8, 7, 0, 1, 8, 2, 3 };
};

template <>
struct HierarchicalShape<7, 2> {
    static constexpr int post[18] = { 7, 0, 1, 8, 7, 2, 9, 3, 4, 10, 8, 9, 11, 10, 5, 12, 11, 6 };
    static constexpr int pre[18] = { 12, 11, 6, 11, 10, 5, 10, 8, 9, 8, 7, 2, 7, 0, 1, 9, 3, 4 };
};

template <>
struct HierarchicalShape<7, 3> {
    static constexpr int post[18] = { 7, 0, 1, 8, 7, 2, 9, 8, 3, 10, 4, 5, 11, 9, 10, 12, 11, 6 };
    static constexpr int pre[18] = { 12, 11, 6, 11, 9, 10, 9, 8, 3, 8, 7, 2, 7, 0, 1, 10, 4, 5 };
};

template <>
struct HierarchicalShape<7, 4> {
    static constexpr int post[18] = { 7, 0, 1, 8, 7, 2, 9, 8, 3, 10, 9, 4, 11, 5, 6, 12, 10, 11 };
    static constexpr int pre[18] = { 12, 10, 11, 10, 9, 4, 9, 8, 3, 8, 7, 2, 7, 0, 1, 11, 5, 6 };
};

template <>
struct HierarchicalShape<7, 5> {
    static constexpr int post[18] = { 7, 0, 1, 8, 2, 3, 9, 7, 8, 10, 4, 5, 11, 9, 10, 12, 11, 6 };
    static constexpr int pre[18] = { 12, 11, 6, 11, 9, 10, 9, 7, 8, 7, 0, 1, 8, 2, 3, 10, 4, 5 };
};

template <>
struct HierarchicalShape<7, 6> {
    static constexpr int post[18] = { 7, 0, 1, 8, 2, 3, 9, 7, 8, 10, 9, 4, 11, 5, 6, 12, 10, 11 };
    static constexpr int pre[18] = { 12, 10, 11, 10, 9, 4, 9, 7, 8, 7, 0, 1, 8, 2, 3, 11, 5, 6 };
};

template <>
struct HierarchicalShape<7, 7> {
    static constexpr int post[18] = { 7, 0, 1, 8, 7, 2, 9, 3, 4, 10, 9, 5, 11, 8, 10, 12, 11, 6 };
    static constexpr int pre[18] = { 12, 11, 6, 11, 8, 10, 8, 7, 2, 7, 0, 1, 10, 9, 5, 9, 3, 4 };
};

template <>
struct HierarchicalShape<7, 8> {
    static constexpr int post[18] = { 7, 0, 1, 8, 7, 2, 9, 3, 4, 10, 8, 9, 11, 5, 6, 12, 10, 11 };
    static constexpr int pre[18] = { 12, 10, 11, 10, 8, 9, 8, 7, 2, 7, 0, 1, 9, 3, 4, 11, 5, 6 };
};

template <>
struct HierarchicalShape<7, 9> {
    static constexpr int post[18] = { 7, 0, 1, 8, 7, 2, 9, 8, 3, 10, 4, 5, 11, 10, 6, 12, 9, 11 };
    static constexpr int pre[18] = { 12, 9, 11, 9, 8, 3, 8, 7, 2, 7, 0, 1, 11, 10, 6, 10, 4, 5 };
};

template <>
struct HierarchicalShape<7, 10> {
    static constexpr int post[18] = { 7, 0, 1, 8, 2, 3, 9, 7, 8, 10, 4, 5, 11, 10, 6, 12, 9, 11 };
    static constexpr int pre[18] = { 12, 9, 11, 9, 7, 8, 7, 0, 1, 8, 2, 3, 11, 10, 6, 10, 4, 5 };
};

template <>
struct HierarchicalShapeCount<8> {
    static constexpr int value = 23;
};

template <>
struct HierarchicalShape<8, 0> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 10, 4, 12, 11, 5, 13, 12, 6, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 12, 6, 12, 11, 5, 11, 10, 4, 10, 9, 3, 9, 8, 2, 8, 0, 1 };
};

template <>
struct HierarchicalShape<8, 1> {
    static constexpr int post[21] = { 8, 0, 1, 9, 2, 3, 10, 8, 9, 11, 10, 4, 12, 11, 5, 13, 12, 6, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 12, 6, 12, 11, 5, 11, 10, 4, 10, 8, 9, 8, 0, 1, 9, 2, 3 };
};

template <>
struct HierarchicalShape<8, 2> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 3, 4, 11, 9, 10, 12, 11, 5, 13, 12, 6, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 12, 6, 12, 11, 5, 11, 9, 10, 9, 8, 2, 8, 0, 1, 10, 3, 4 };
};

template <>
struct HierarchicalShape<8, 3> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 4, 5, 12, 10, 11, 13, 12, 6, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 12, 6, 12, 10, 11, 10, 9, 3, 9, 8, 2, 8, 0, 1, 11, 4, 5 };
};

template <>
struct HierarchicalShape<8, 4> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 10, 4, 12, 5, 6, 13, 11, 12, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 11, 12, 11, 10, 4, 10, 9, 3, 9, 8, 2, 8, 0, 1, 12, 5, 6 };
};

template <>
struct HierarchicalShape<8, 5> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 10, 4, 12, 11, 5, 13, 6, 7, 14, 12, 13 };
    static constexpr int pre[21] = { 14, 12, 13, 12, 11, 5, 11, 10, 4, 10, 9, 3, 9, 8, 2, 8, 0, 1, 13, 6, 7 };
};

template <>
struct HierarchicalShape<8, 6> {
    static constexpr int post[21] = { 8, 0, 1, 9, 2, 3, 10, 8, 9, 11, 4, 5, 12, 10, 11, 13, 12, 6, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 12, 6, 12, 10, 11, 10, 8, 9, 8, 0, 1, 9, 2, 3, 11, 4, 5 };
};

template <>
struct HierarchicalShape<8, 7> {
    static constexpr int post[21] = { 8, 0, 1, 9, 2, 3, 10, 8, 9, 11, 10, 4, 12, 5, 6, 13, 11, 12, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 11, 12, 11, 10, 4, 10, 8, 9, 8, 0, 1, 9, 2, 3, 12, 5, 6 };
};

template <>
struct HierarchicalShape<8, 8> {
    static constexpr int post[21] = { 8, 0, 1, 9, 2, 3, 10, 8, 9, 11, 10, 4, 12, 11, 5, 13, 6, 7, 14, 12, 13 };
    static constexpr int pre[21] = { 14, 12, 13, 12, 11, 5, 11, 10, 4, 10, 8, 9, 8, 0, 1, 9, 2, 3, 13, 6, 7 };
};

template <>
struct HierarchicalShape<8, 9> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 3, 4, 11, 10, 5, 12, 9, 11, 13, 12, 6, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 12, 6, 12, 9, 11, 9, 8, 2, 8, 0, 1, 11, 10, 5, 10, 3, 4 };
};

template <>
struct HierarchicalShape<8, 10> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 3, 4, 11, 9, 10, 12, 5, 6, 13, 11, 12, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 11, 12, 11, 9, 10, 9, 8, 2, 8, 0, 1, 10, 3, 4, 12, 5, 6 };
};

template <>
struct HierarchicalShape<8, 11> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 3, 4, 11, 9, 10, 12, 11, 5, 13, 6, 7, 14, 12, 13 };
    static constexpr int pre[21] = { 14, 12, 13, 12, 11, 5, 11, 9, 10, 9, 8, 2, 8, 0, 1, 10, 3, 4, 13, 6, 7 };
};

template <>
struct HierarchicalShape<8, 12> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 4, 5, 12, 11, 6, 13, 10, 12, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 10, 12, 10, 9, 3, 9, 8, 2, 8, 0, 1, 12, 11, 6, 11, 4, 5 };
};

template <>
struct HierarchicalShape<8, 13> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 4, 5, 12, 10, 11, 13, 6, 7, 14, 12, 13 };
    static constexpr int pre[21] = { 14, 12, 13, 12, 10, 11, 10, 9, 3, 9, 8, 2, 8, 0, 1, 11, 4, 5, 13, 6, 7 };
};

template <>
struct HierarchicalShape<8, 14> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 10, 4, 12, 5, 6, 13, 12, 7, 14, 11, 13 };
    static constexpr int pre[21] = { 14, 11, 13, 11, 10, 4, 10, 9, 3, 9, 8, 2, 8, 0, 1, 13, 12, 7, 12, 5, 6 };
};

template <>
struct HierarchicalShape<8, 15> {
    static constexpr int post[21] = { 8, 0, 1, 9, 2, 3, 10, 8, 9, 11, 4, 5, 12, 11, 6, 13, 10, 12, 14, 13, 7 };
    static constexpr int pre[21] = { 14, 13, 7, 13, 10, 12, 10, 8, 9, 8, 0, 1, 9, 2, 3, 12, 11, 6, 11, 4, 5 };
};

template <>
struct HierarchicalShape<8, 16> {
    static constexpr int post[21] = { 8, 0, 1, 9, 2, 3, 10, 8, 9, 11, 4, 5, 12, 10, 11, 13, 6, 7, 14, 12, 13 };
    static constexpr int pre[21] = { 14, 12, 13, 12, 10, 11, 10, 8, 9, 8, 0, 1, 9, 2, 3, 11, 4, 5, 13, 6, 7 };
};

template <>
struct HierarchicalShape<8, 17> {
    static constexpr int post[21] = { 8, 0, 1, 9, 2, 3, 10, 8, 9, 11, 10, 4, 12, 5, 6, 13, 12, 7, 14, 11, 13 };
    static constexpr int pre[21] = { 14, 11, 13, 11, 10, 4, 10, 8, 9, 8, 0, 1, 9, 2, 3, 13, 12, 7, 12, 5, 6 };
};

template <>
struct HierarchicalShape<8, 18> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 3, 4, 11, 10, 5, 12, 9, 11, 13, 6, 7, 14, 12, 13 };
    static constexpr int pre[21] = { 14, 12, 13, 12, 9, 11, 9, 8, 2, 8, 0, 1, 11, 10, 5, 10, 3, 4, 13, 6, 7 };
};

template <>
struct HierarchicalShape<8, 19> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 3, 4, 11, 9, 10, 12, 5, 6, 13, 12, 7, 14, 11, 13 };
    static constexpr int pre[21] = { 14, 11, 13, 11, 9, 10, 9, 8, 2, 8, 0, 1, 10, 3, 4, 13, 12, 7, 12, 5, 6 };
};

template <>
struct HierarchicalShape<8, 20> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 4, 5, 12, 11, 6, 13, 12, 7, 14, 10, 13 };
    static constexpr int pre[21] = { 14, 10, 13, 10, 9, 3, 9, 8, 2, 8, 0, 1, 13, 12, 7, 12, 11, 6, 11, 4, 5 };
};

template <>
struct HierarchicalShape<8, 21> {
    static constexpr int post[21] = { 8, 0, 1, 9, 8, 2, 10, 9, 3, 11, 4, 5, 12, 6, 7, 13, 11, 12, 14, 10, 13 };
    static constexpr int pre[21] = { 14, 10, 13, 10, 9, 3, 9, 8, 2, 8, 0, 1, 13, 11, 12, 11, 4, 5, 12, 6, 7 };
};

template <>
struct HierarchicalShape<8, 22> {
    static constexpr int post[21] = { 8, 0, 1, 9, 2, 3, 10, 8, 9, 11, 4, 5, 12, 6, 7, 13, 11, 12, 14, 10, 13 };
    static constexpr int pre[21] = { 14, 10, 13, 10, 8, 9, 8, 0, 1, 9, 2, 3, 13, 11, 12, 11, 4, 5, 12, 6, 7 };
};

inline int hierarchicalShapeId(int stateCount, const char* code) {
    static const char* const codes[] = {
        "(LL)",
        "((LL)L)",
        "(((LL)L)L)",
        "((LL)(LL))",
        "((((LL)L)L)L)",
        "(((LL)(LL))L)",
        "(((LL)L)(LL))",
        "(((((LL)L)L)L)L)",
        "((((LL)(LL))L)L)",
        "((((LL)L)(LL))L)",
        "((((LL)L)L)(LL))",
        "(((LL)(LL))(LL))",
        "(((LL)L)((LL)L))",
        "((((((LL)L)L)L)L)L)",
        "(((((LL)(LL))L)L)L)",
        "(((((LL)L)(LL))L)L)",
        "(((((LL)L)L)(LL))L)",
        "(((((LL)L)L)L)(LL))",
        "((((LL)(LL))(LL))L)",
        "((((LL)(LL))L)(LL))",
        "((((LL)L)((LL)L))L)",
        "((((LL)L)(LL))(LL))",
        "((((LL)L)L)((LL)L))",
        "(((LL)(LL))((LL)L))",
        "(((((((LL)L)L)L)L)L)L)",
        "((((((LL)(LL))L)L)L)L)",
        "((((((LL)L)(LL))L)L)L)",
        "((((((LL)L)L)(LL))L)L)",
        "((((((LL)L)L)L)(LL))L)",
        "((((((LL)L)L)L)L)(LL))",
        "(((((LL)(LL))(LL))L)L)",
        "(((((LL)(LL))L)(LL))L)",
        "(((((LL)(LL))L)L)(LL))",
        "(((((LL)L)((LL)L))L)L)",
        "(((((LL)L)(LL))(LL))L)",
        "(((((LL)L)(LL))L)(LL))",
        "(((((LL)L)L)((LL)L))L)",
        "(((((LL)L)L)(LL))(LL))",
        "(((((LL)L)L)L)((LL)L))",
        "((((LL)(LL))((LL)L))L)",
        "((((LL)(LL))(LL))(LL))",
        "((((LL)(LL))L)((LL)L))",
        "((((LL)L)((LL)L))(LL))",
        "((((LL)L)(LL))((LL)L))",
        "((((LL)L)L)(((LL)L)L))",
        "((((LL)L)L)((LL)(LL)))",
        "(((LL)(LL))((LL)(LL)))"
    };
    static const int first[] = { 0, 1, 2, 4, 7, 13, 24, 47 };
    if (stateCount < 2 || stateCount > kHierarchicalShapeTableMaxStates)
        return -1;
    for (int i = first[stateCount - 2]; i < first[stateCount - 1]; i++)
        if (std::strcmp(codes[i], code) == 0)
            return i - first[stateCount - 2];
    return -1;
}

}
}

#endif
