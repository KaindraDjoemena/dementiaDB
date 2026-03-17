#include "metadata.h"
#include "nlohmann/json_fwd.hpp"

#include <string>


namespace demDB
{

// PUBLIC
void Metadata::insert(nlohmann::json r)
{
    m_records.push_back(std::move(r));
}

void Metadata::insertBulk(std::vector<nlohmann::json>&& records)
{
    m_records.insert(
            m_records.end(), 
            std::make_move_iterator(records.begin()), 
            std::make_move_iterator(records.end())
        );
}

void Metadata::remove(size_t id)
{
    m_records[id] = nullptr;    // soft delete records so that index ids dont get misaligned with metadata ids
}

void Metadata::save(std::ofstream& f)
{
    // Size of m_records
    size_t recordsSize = m_records.size();
    f.write((const char*)&recordsSize, sizeof(recordsSize));

    // Records
    for (nlohmann::json& record : m_records)
    {
        std::vector<uint8_t> v = nlohmann::json::to_msgpack(record);
        
        size_t blobSize = v.size();
        f.write((const char*)&blobSize, sizeof(blobSize));
        
        f.write((const char*)v.data(), v.size());
    }
}

const nlohmann::json& Metadata::get(size_t id) const
{
    return m_records.at(id);
}

void Metadata::open(std::ifstream& f)
{
    // Size of m_records
    size_t recordsSize;
    if (!f.read((char*)&recordsSize, sizeof(recordsSize))) return;

    // Records
    m_records.clear();
    m_records.reserve(recordsSize);
    for (size_t i = 0; i < recordsSize; i++)
    {
        size_t blobSize;
        f.read((char*)&blobSize, sizeof(blobSize));

        std::vector<uint8_t> v(blobSize);
        f.read((char*)v.data(), blobSize);

        m_records.push_back(nlohmann::json::from_msgpack(v));
    }
}

} /* namespace demDB */