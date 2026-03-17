#include "dementiaDB.h"

#include "index.h"
#include "indexFlat.h"
#include "indexHnsw.h"
#include "metadata.h"

#include "nlohmann/json_fwd.hpp"
#include <cstdio>
#include <memory>
#include <stdexcept>


namespace demDB
{

void DementiaDB::bulkLoad(const std::string& colName, const std::filesystem::path& hdf5Path, const std::string& datasetName, const std::filesystem::path& metadataPath)
{
    auto& col = m_collections.at(colName);
    if (!col.index->bulkInsert(hdf5Path, datasetName))
    {
        throw std::runtime_error("Failed to load HDF5 vectors");
    }

    std::ifstream f(metadataPath);
    if (!f.is_open())
    {
        throw std::runtime_error("Could not open metadata file: " + metadataPath.string());
    }
    nlohmann::json rawMeta = nlohmann::json::parse(f);

    size_t numVecs = col.index->size();
    std::vector<nlohmann::json> flatMetadata;
    
    flatMetadata.reserve(numVecs);

    for (size_t i = 0; i < numVecs; ++i)
    {
        std::string key = std::to_string(i);
        if (rawMeta.contains(key))
        {
            flatMetadata.push_back(rawMeta[key]);
        }
        else
        {
            flatMetadata.push_back({{"error", "missing metadata"}});
        }
    }
    
    col.metadata.insertBulk(std::move(flatMetadata));
}

void DementiaDB::createCollection(const std::string& colName, INDEX_TYPE type)
{
    Collection newCollection;
    newCollection.type = type;

    switch (type)
    {
        case INDEX_TYPE::FLAT:
            newCollection.index = std::make_unique<IndexFlat>();
            break;
        
        case INDEX_TYPE::HNSW:
            newCollection.index = std::make_unique<IndexHNSW>();
            break;
    }

    m_collections[colName] = std::move(newCollection);
}

void DementiaDB::setIndexMetric(const std::string& colName, METRIC metric)
{
    auto it = m_collections.find(colName);
    if (it == m_collections.end())
    {
        throw std::runtime_error("Collection '" + colName + "' does not exist");
    }

    it->second.index->setMetric(metric);
}

METRIC DementiaDB::getIndexMetric(const std::string& colName) const
{
    auto it = m_collections.find(colName);
    if (it == m_collections.end())
    {
        throw std::runtime_error("Collection '" + colName + "' does not exist");
    }

    return it->second.index->getMetric();
}

void DementiaDB::setIndexHNSWParam(const std::string& colName, size_t value, HNSW_PARAM param)
{
    auto it = m_collections.find(colName);
    if (it == m_collections.end())
    {
        throw std::runtime_error("Collection '" + colName + "' does not exist");
    }

    auto* hnsw = dynamic_cast<IndexHNSW*>(it->second.index.get());
    if (!hnsw)
    {
        throw std::runtime_error("Collection '" + colName + "' index is not of type IndexHNSW");
    }

    switch (param)
    {
        case HNSW_PARAM::EF_CONSTRUCT:
            hnsw->setEfConstruction(value);
            break;
        case HNSW_PARAM::EF:
            hnsw->setEf(value);
            break;
        case HNSW_PARAM::M:
            hnsw->setM(value);
            break;
        case HNSW_PARAM::M_MAX:
            hnsw->setMmax(value);
            break;
        case HNSW_PARAM::M_MAX0:
            hnsw->setMmax0(value);
            break;
        default:
            throw std::invalid_argument("Unknown HNSW_PARAM");
    }
}

size_t DementiaDB::getIndexHNSWParam(const std::string& colName, HNSW_PARAM param) const
{
    auto it = m_collections.find(colName);
    if (it == m_collections.end())
    {
        throw std::runtime_error("Collection '" + colName + "' does not exist");
    }

    auto* hnsw = dynamic_cast<IndexHNSW*>(it->second.index.get());
    if (!hnsw)
    {
        throw std::runtime_error("Collection '" + colName + "' index is not of type IndexHNSW");
    }

    switch (param)
    {
        case HNSW_PARAM::EF_CONSTRUCT:
            return hnsw->getEfConstruction();
        case HNSW_PARAM::EF:
            return hnsw->getEf();
        case HNSW_PARAM::M:
            return hnsw->getM();
        case HNSW_PARAM::M_MAX:
            return hnsw->getMmax();
        case HNSW_PARAM::M_MAX0:
            return hnsw->getMmax0();
        default:
            throw std::invalid_argument("Unknown HNSW_PARAM");
    }
}

std::vector<nlohmann::json> DementiaDB::search(const std::string& colName, const std::vector<float>& q, size_t k)
{
    auto it = m_collections.find(colName);
    if (it == m_collections.end())
    {
        return {};
    }

    std::vector<Index::Candidate> searchResults = it->second.index->search(q.data(), k);

    std::vector<nlohmann::json> results;
    results.reserve(searchResults.size());

    for (const auto& candidate : searchResults)
    {
        results.push_back(it->second.metadata.get(candidate.id));
    }

    return results;
}

void DementiaDB::insert(const std::string& colName, const float* q, const nlohmann::json& metadata)
{
    auto it = m_collections.find(colName);
    if (it == m_collections.end())
    {
        throw std::runtime_error("Collection '" + colName + "' does not exist");
    }

    it->second.index->insert(q);
    it->second.metadata.insert(metadata);
}

void DementiaDB::bulkInsert(const std::string& colName, const float* data, size_t numVecs, size_t numDims, std::vector<nlohmann::json>&& metadata)
{
    auto& col = m_collections.at(colName);

    col.index->bulkInsert(data, numVecs, numDims);
    col.metadata.insertBulk(std::move(metadata));
}

void DementiaDB::remove(const std::string& colName, size_t id)
{
    auto it = m_collections.find(colName);
    if (it == m_collections.end())
    {
        throw std::runtime_error("Collection '" + colName + "' does not exist");
    }

    m_collections[colName].index->softDelete(id);
    m_collections[colName].metadata.remove(id);
}

int DementiaDB::save(const std::filesystem::path& p)
{
    try
    {
        std::ofstream f(p, std::ios::binary);
        f.exceptions(std::ofstream::failbit | std::ofstream::badbit);

        size_t collectionsSize = m_collections.size();
        f.write((const char*)&collectionsSize, sizeof(collectionsSize));
        for (auto& [name, collection] : m_collections)
        {
            // Collection Name
            size_t nameLen = name.size();
            f.write((const char*)&nameLen, sizeof(nameLen));
            f.write(name.c_str(), nameLen);

            // Collection Type
            f.write((const char*)&collection.type, sizeof(collection.type));

            collection.index->save(f);
            collection.metadata.save(f);
        }

        f.close();
        
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }

}

int DementiaDB::open(const std::filesystem::path& p)
{
    try
    {
        if (!std::filesystem::exists(p)) return -1;
        
        std::ifstream f(p, std::ios::binary);
        f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        
        std::unordered_map<std::string, Collection> tempCollection;
        
        size_t collectionsSize;
        f.read((char*)&collectionsSize, sizeof(collectionsSize));
        tempCollection.reserve(collectionsSize);
        for (size_t i = 0; i < collectionsSize; i++)
        {
            // Collection Name
            size_t nameLen;
            f.read((char*)&nameLen, sizeof(nameLen));
            std::string colName(nameLen, '\0');
            f.read(colName.data(), nameLen);
    
            // Collection Type
            INDEX_TYPE type;
            f.read((char*)&type, sizeof(type));
    
            Collection newCollection;
            switch (type)
            {
                case INDEX_TYPE::FLAT:
                    newCollection.index = std::make_unique<IndexFlat>();
                    break;
                case INDEX_TYPE::HNSW:
                    newCollection.index = std::make_unique<IndexHNSW>();
                    break;
            }
    
            newCollection.type = type;
            newCollection.index->open(f);
            newCollection.metadata.open(f);
    
            tempCollection[colName] = std::move(newCollection);
        }
    
        f.close();
    
        m_collections = std::move(tempCollection);

        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}

} /* namespace demDB */