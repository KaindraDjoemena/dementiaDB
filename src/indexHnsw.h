/*
 * The HNSW data structure, algorithms, and parameters are based on the paper:
 * "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs",
 * by Yu. A. Malkov & D. A. Yashunin
 *
 * link: https://arxiv.org/pdf/1603.09320
*/

#pragma once

#include "index.h"

#include <fstream>
#include <random>
#include <mutex>
#include <shared_mutex>
#include <atomic>


namespace demDB
{

class IndexHNSW : public Index
{
public:
    IndexHNSW();
    
    void setEfConstruction(size_t efConstruct) { m_efConstruct = efConstruct; }
    void setEf(size_t ef) { m_ef = ef; }
    void setM(size_t M) { m_M = M; }
    void setMmax(size_t Mmax) { m_Mmax = Mmax; }
    void setMmax0(size_t Mmax0) { m_Mmax0 = Mmax0; }

    size_t getEfConstruction() const { return m_efConstruct; }
    size_t getEf() const { return m_ef; }
    size_t getM() const { return m_M; }
    size_t getMmax() const { return m_Mmax; }
    size_t getMmax0() const { return m_Mmax0; }

    void build() override;

    size_t insert(const float* q) override;

    void bulkInsert(const float* data, size_t numVecs, size_t numDims) override;
    
    std::vector<Candidate> search(const float* q, size_t k) const override;
    
    void save(std::ofstream& p) override;
    
    void open(std::ifstream& p) override;
    
private:
    std::vector<int> m_levels;
    std::vector<std::shared_ptr<std::shared_mutex>> m_nodeLocks;
    
    struct Node
    {
        int maxLayer;
        std::vector<std::vector<int>> neighbors;    // node neighbors at every layer
    };
    std::vector<Node> m_nodes;
    
    std::atomic<int> m_maxLayer{-1};
    std::atomic<int> m_entryPoint{-1};
    
    size_t m_efConstruct = 100;     // higher = slower construction, faster search 
    size_t m_ef          = 100;     // search quality
    
    // M closest neighbors at each layer
    size_t m_M     = 24;
    size_t m_Mmax  = 24;
    size_t m_Mmax0 = 2 * m_M;
    
    // Normalization for factor for level generation
    float m_mL = 1.0f / std::log(m_M);
    
    mutable std::mt19937 m_rng;


    void insertNode(size_t id);

    std::vector<Candidate> searchLayer(const float* q, const std::vector<int>& ep, size_t ef, int lc) const;

    std::vector<Candidate> selectNeighbors(const float* q, const std::vector<Candidate>& candidates, size_t M, int lc, bool extendCandidates) const;
    
    int generateLevel() const;
};

} /* namespace demDB */