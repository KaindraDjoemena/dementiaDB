#pragma once

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include <fstream>
#include <vector>


namespace demDB
{

class Metadata
{
public:
    void insert(nlohmann::json record);

    void insertBulk(std::vector<nlohmann::json>&& records);
    
    void remove(size_t id);
    
    const nlohmann::json& get(size_t id) const;
    
    void save(std::ofstream& f);

    void open(std::ifstream& f);

private:
    std::vector<nlohmann::json> m_records;
};

} /* namespace demDB */