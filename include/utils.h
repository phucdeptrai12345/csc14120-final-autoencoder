#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

struct Tensor
{
    int c, h, w;
    std::vector<float> data;

    // Constructor
    Tensor(int _c = 1, int _h = 1, int _w = 1) : c(_c), h(_h), w(_w)
    {
        data.resize(c * h * w, 0.0f);
    }


    float &operator()(int ch, int row, int col)
    {
        return data[ch * h * w + row * w + col];
    }

    const float &operator()(int ch, int row, int col) const
    {
        return data[ch * h * w + row * w + col];
    }

    void zero_grad()
    {
        std::fill(data.begin(), data.end(), 0.0f);
    }
};

// create random weights
inline void initialize_weights(std::vector<float> &weights, int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 0.02f);
    for (int i = 0; i < size; ++i)
    {
        weights[i] = d(gen);
    }
}

#endif // UTILS_H