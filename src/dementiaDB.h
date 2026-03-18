#pragma once

#include "index.h"
#include "metadata.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <memory>


namespace demDB
{

enum class INDEX_TYPE : uint32_t
{
    FLAT,
    HNSW
};

enum class HNSW_PARAM : uint32_t
{
    EF_CONSTRUCT,
    EF,
    M,
    M_MAX,
    M_MAX0
};


class DementiaDB
{
public:
    void bulkLoad(const std::string& colName, const std::filesystem::path& hdf5Path, const std::string& datasetName, const std::filesystem::path& metadataPath);

    void createCollection(const std::string& colName, INDEX_TYPE type);

    void setIndexMetric(const std::string& colName, METRIC metric);

    METRIC getIndexMetric(const std::string& colName) const;

    void setIndexHNSWParam(const std::string& colName, size_t value, HNSW_PARAM param);

    size_t getIndexHNSWParam(const std::string& colName, HNSW_PARAM param) const;

    std::vector<nlohmann::json> search(const std::string& colName, const std::vector<float>& q, size_t k, const nlohmann::json& filter={}, size_t searchMult=3);
    
    size_t insert(const std::string& colName, const float* q, const nlohmann::json& metadata);

    void bulkInsert(const std::string& colName, const float* data, size_t numVecs, size_t numDims, std::vector<nlohmann::json>&& metadata);

    void remove(const std::string& colName, size_t id);

    int save(const std::filesystem::path& p);

    int open(const std::filesystem::path& p);

private:
    struct Collection {
        INDEX_TYPE type;
        std::unique_ptr<Index> index;
        Metadata metadata;
    };
    std::unordered_map<std::string, Collection> m_collections;

    bool postFilter(const nlohmann::json& meta, const nlohmann::json& filter);
};

} /* namespace demDB */