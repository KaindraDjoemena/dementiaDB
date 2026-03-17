#include "index.h"

#include "../utils/mathutils.h"

#include <highfive/highfive.hpp>
#include <unordered_set>


namespace demDB
{

void Index::setMetric(METRIC metric)
{
    m_metric = metric;
    switch (metric)
    {
        case METRIC::L2:
            m_distFunc = L2Square;
            break;
        case METRIC::COSINE:
            m_distFunc = [](const float* a, const float* b, size_t n) { return 1.0f - cosine(a, b, n); };
            break;
        case METRIC::NORM_COSINE:
            m_distFunc = [](const float* a, const float* b, size_t n) { return 1.0f - dot(a, b, n); };
            break;
        case METRIC::INNER_PROD:
            m_distFunc = [](const float* a, const float* b, size_t n) { return -dot(a, b, n); };
            break;
    }
}

const float* Index::getVector(size_t id) const
{
    return m_vectors.data() + id * m_numDimensions;
}

void Index::softDelete(size_t id)
{
    m_deletedVecs.insert(id);
}

size_t Index::size() const
{
    return m_numVectors;
}

bool Index::bulkInsert(const std::filesystem::path& p, const std::string& setName)
{
    try
    {
        std::cout << "INDEX::BULK_INSERT: " << p.string() << std::endl;

        HighFive::File file(p.string(), HighFive::File::ReadOnly);
        HighFive::DataSet dataset = file.getDataSet(setName);
    
        std::vector<size_t> dims = dataset.getDimensions();
        size_t numNewVecs    = dims[0];
        size_t numNewDims    = dims[1];

        if (m_numVectors > 0 && numNewDims != m_numDimensions)
        {
            std::cerr << "INDEX::ERROR: Dimension mismatch. DB has " << m_numDimensions 
                      << " but HDF5 has " << numNewDims << std::endl;
            return false;
        }

        std::vector<float> newData(numNewVecs * numNewDims);
        dataset.read(newData.data());

        bulkInsert(newData.data(), numNewVecs, numNewDims);

        return true;
    }
    catch (const std::exception& err)
    {
        std::cerr << "INDEX::BULK_INSERT_ERROR: " << err.what() << std::endl;
        return false;
    }
}
    
bool Index::loadFromHDF5(const std::filesystem::path& p, const std::string& setName)
{
    try
    {
        std::cout << "INDEX::LOADFROMHD5: " << p.string() << std::endl;

        HighFive::File file(p.string(), HighFive::File::ReadOnly);

        HighFive::DataSet dataset = file.getDataSet(setName);
    
        std::vector<size_t> dims = dataset.getDimensions();
        m_numVectors    = dims[0];
        m_numDimensions = dims[1];
    
        m_vectors.resize(m_numVectors * m_numDimensions);
    
        dataset.read(m_vectors.data());

        return true;
    }
    catch (const HighFive::Exception& err)
    {
        std::cerr << "INDEX::LOADFROMHD5 " << err.what() << std::endl;

        return false;
    }    
}

float Index::d(const float* a, const float* b, size_t size) const
{
    return m_distFunc(a, b, size);
}

float Index::d(const std::vector<float>& a, const std::vector<float>& b) const
{
    return d(a.data(), b.data(), a.size());
}

} /* namespace demDB */