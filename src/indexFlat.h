#pragma once

#include "index.h"

#include <filesystem>


namespace demDB
{

class IndexFlat : public Index
{
public:
    IndexFlat();
    
    void build() override;
    
    size_t insert(const float* q) override;
    
    void bulkInsert(const float* data, size_t numVecs, size_t numDims) override;

    std::vector<Candidate> search(const float* q, size_t k) const override;
    
    void save(std::ofstream& f) override;
    
    void open(std::ifstream& f) override;

};

} /* namespace debDB */