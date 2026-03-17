#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <unordered_set>


namespace demDB
{

enum class METRIC : uint32_t
{
    L2,
    COSINE,
    NORM_COSINE,
    INNER_PROD
};

class Index
{
public:
    struct Candidate
    {
        size_t id;
        float distance;

        bool operator < (const Candidate& other) const
        {
            return distance < other.distance;
        }

        bool operator > (const Candidate& other) const
        {
            return distance > other.distance;
        }
    };

public:
    void setMetric(METRIC metric);

    METRIC getMetric() const { return m_metric; }

    virtual ~Index() = default;
    
    virtual void build() = 0;
    
    virtual size_t insert(const float* q) = 0;

    virtual void bulkInsert(const float* data, size_t numVecs, size_t numDims) = 0;
    
    bool bulkInsert(const std::filesystem::path& p, const std::string& setName);
    
    virtual std::vector<Candidate> search(const float* q, size_t k) const = 0;
    
    virtual void save(std::ofstream& f) = 0;
    
    virtual void open(std::ifstream& p) = 0;
    
    const float* getVector(size_t id) const;
    
    void softDelete(size_t id);
    
    size_t size() const;

    bool loadFromHDF5(const std::filesystem::path& p, const std::string& setName);

protected:
    size_t m_numVectors;
    size_t m_numDimensions;

    METRIC m_metric = METRIC::L2;
    
    std::vector<float> m_vectors;    
    std::unordered_set<size_t> m_deletedVecs;

    using DistFunc = float(*)(const float*, const float*, size_t);
    DistFunc m_distFunc = nullptr;

    float d(const float* a, const float* b, size_t size) const;

    float d(const std::vector<float>& a, const std::vector<float>& b) const;
};

} /* namespace demDB */