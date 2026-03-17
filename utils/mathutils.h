#pragma once

#include <vector>
#include <string>
#include <numeric>
#include <cmath>


namespace demDB
{

inline
float L2Square(const float* __restrict__ a, const float* __restrict__ b, size_t size)
{
    float total = 0;
    for (size_t i = 0; i < size; i++)
    {
        float d  = a[i] - b[i];
        total   += d * d;
    }
    return total;
}

inline
float L2Square(const std::vector<float>& a, const std::vector<float>& b)
{
   return L2Square(a.data(), b.data(), a.size());
}

inline
float squareLen(const float* __restrict__ a, size_t size)
{
   float sum = 0.0f;
   for (size_t i = 0; i < size; i++)
   {
       sum += a[i] * a[i];
   }
   return sum;
}

inline
float squareLen(const std::vector<float>& a)
{
   return squareLen(a.data(), a.size());
}

inline
float dot(const float* __restrict__ a, const float* __restrict__ b, size_t size)
{
   float sum = 0.0f;
   for (size_t i = 0; i < size; i++)
   {
       sum += a[i] * b[i];
   }
   return sum;
}

inline
float dot(const std::vector<float>& a, const std::vector<float>& b)
{
   return dot(a.data(), b.data(), a.size());
}

inline
float mag(const float* a, size_t size)
{
   return std::sqrt(squareLen(a, size));
}

inline
float mag(const std::vector<float>& a)
{
   return mag(a.data(), a.size());
}

inline
float cosine(const float* __restrict__ a, const float* __restrict__ b, size_t size)
{
   float dotProd = dot(a, b, size);
   float magA = mag(a, size);
   float magB = mag(b, size);
   return dotProd / (magA * magB);
}

inline
float cosine(const std::vector<float>& a, const std::vector<float>& b)
{
   return cosine(a.data(), b.data(), a.size());
}

} /* namespace demDB */