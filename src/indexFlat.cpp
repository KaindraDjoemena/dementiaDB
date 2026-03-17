#include "indexFlat.h"

#include "../utils/mathutils.h"

#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <highfive/highfive.hpp>


namespace demDB
{

// PUBLIC
IndexFlat::IndexFlat()
{
    m_numVectors    = 0;
    m_numDimensions = 0;
}

void IndexFlat::build()
{
}

size_t IndexFlat::insert(const float* q)
{
    m_vectors.insert(m_vectors.end(), q, q + m_numDimensions);
    
    size_t newID = m_numVectors;
    m_numVectors++;
    
    return newID;
}

void IndexFlat::bulkInsert(const float* data, size_t numVecs, size_t numDims)
{
    if (numVecs == 0)
    {
        return;
    }

    if (m_numVectors == 0)
    {
        m_numDimensions = numDims;
    }
    else if (m_numDimensions != numDims)
    {
        throw std::runtime_error("IndexFlat::bulkInsert: dimension mismatch");
    }

    m_vectors.reserve((m_numVectors + numVecs) * m_numDimensions);

    m_vectors.insert(m_vectors.end(), data, data + (numVecs * m_numDimensions));

    m_numVectors += numVecs;
}

std::vector<IndexFlat::Candidate> IndexFlat::search(const float* q, size_t k) const
{
    std::priority_queue<Candidate> topK;

    const float* dataPtr = m_vectors.data();

    for (size_t i = 0; i < m_numVectors; i++)
    {
        if (m_deletedVecs.count(i))
        {
            continue; 
        }

        float d2 = d(q, dataPtr + (i * m_numDimensions), m_numDimensions);
    
        if (topK.size() < k)
        {
            topK.push({i, d2});
        }
        else if (d2 < topK.top().distance)
        {
            topK.pop();
            topK.push({i, d2});
        }
    }

    std::vector<Candidate> topKResults;
    topKResults.reserve(k);

    while (!topK.empty())
    {
        topKResults.emplace_back(topK.top());
        topK.pop();
    }

    std::reverse(topKResults.begin(), topKResults.end());

    return topKResults;
}

void IndexFlat::save(std::ofstream& f)
{
    // Index Distance Function
    f.write((const char*)&m_metric,         sizeof(m_metric));

    // Vector size
    f.write((const char*)&m_numVectors,     sizeof(m_numVectors));
    f.write((const char*)&m_numDimensions,  sizeof(m_numDimensions));

    // Flat Vectors
    f.write((const char*)m_vectors.data(),  sizeof(float) * m_vectors.size());

    // Soft Deleted Vecs
    size_t deletedSize = m_deletedVecs.size();
    f.write((const char*)&deletedSize,      sizeof(deletedSize));
    for (const auto& id : m_deletedVecs)
    {
        f.write((const char*)&id,           sizeof(id));
    }
}

void IndexFlat::open(std::ifstream& f)
{
    // Index Distance Function
    f.read((char*)&m_metric,        sizeof(m_metric));

    // Vector size
    f.read((char*)&m_numVectors,    sizeof(m_numVectors));
    f.read((char*)&m_numDimensions, sizeof(m_numDimensions));

    // Flat Vectors
    m_vectors.resize(m_numVectors * m_numDimensions);
    f.read((char*)m_vectors.data(), sizeof(float) * m_vectors.size());

    // Soft Deleted Vecs
    m_deletedVecs.clear();
    size_t deletedSize;
    f.read((char*)&deletedSize,     sizeof(deletedSize));
    for (size_t i = 0; i < deletedSize; ++i)
    {
        size_t id;
        f.read((char*)&id,          sizeof(id));
        m_deletedVecs.insert(id);
    }
}

} /* namespace debDB */