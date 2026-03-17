#include "../src/dementiaDB.h"
#include "../utils/timer.h"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>


int main()
{
    const std::string colName = "books";

    demDB::DementiaDB db;
    db.createCollection(colName,       demDB::INDEX_TYPE::HNSW);
    db.setIndexMetric(colName,         demDB::METRIC::COSINE);
    db.setIndexHNSWParam(colName, 16,  demDB::HNSW_PARAM::M);
    db.setIndexHNSWParam(colName, 16,  demDB::HNSW_PARAM::M_MAX);
    db.setIndexHNSWParam(colName, 32,  demDB::HNSW_PARAM::M_MAX0);
    db.setIndexHNSWParam(colName, 100, demDB::HNSW_PARAM::EF_CONSTRUCT);

    long long bulkLoadTimeMs = 0;
    {
        ScopeTimer t(bulkLoadTimeMs);

        db.bulkLoad("books", "books.hdf5", "train", "book_metadata.json");
    }

    std::cout << "Bulk Load Time: " << bulkLoadTimeMs << std::endl;

    db.save("test_db.dem");

    return 0;
}