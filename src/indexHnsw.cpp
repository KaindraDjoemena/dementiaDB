/*
 * The HNSW data structure, algorithms, and parameters are based on the paper:
 * "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs",
 * by Yu. A. Malkov & D. A. Yashunin
 *
 * link: https://arxiv.org/pdf/1603.09320
*/

#include "indexHnsw.h"

#include "../utils/mathutils.h"

#include <cmath>
#include <queue>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <fstream>


namespace demDB
{

// PUBLIC
IndexHNSW::IndexHNSW()
{
    m_metric = METRIC::L2;

    m_numVectors    = 0;
    m_numDimensions = 0;

    m_maxLayer.store(-1);
    m_entryPoint.store(-1);

    m_efConstruct = 100; 
    m_ef          = 100;
    
    m_M     = 24;
    m_Mmax  = 24;
    m_Mmax0 = 2 * m_M;

    m_mL = 1.0f / std::log(m_M);
}

void IndexHNSW::build()
{
    std::cout << "IndexHNSW::build: building index for " << m_numVectors << " vectors of " << m_numDimensions << " dims" << std::endl;
    
    // Pre allocate node levels
    std::cout << "IndesHNSW::build:: allocating node levels" << std::endl;
    m_levels.resize(m_numVectors);
    for (size_t i = 0; i < m_numVectors; i++)
    {
        m_levels[i] = generateLevel();
    }

    // Pre allocate mutexes
    std::cout << "IndexHNSW::build: allocating mutexes" << std::endl;
    m_nodes.reserve(m_numVectors);
    m_nodeLocks.reserve(m_numVectors);
    for (size_t i = 0; i < m_numVectors; i++)
    {
        m_nodes.push_back({m_levels[i], std::vector<std::vector<int>>(m_levels[i] + 1)});
        m_nodeLocks.emplace_back(std::make_shared<std::shared_mutex>());
    }

    // Parallel insert
    std::cout << "IndexHNSW::build: inserting nodes" << std::endl;
    insertNode(0);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 1; i < m_numVectors; i++)
    {
        insertNode(i);
    }
}

size_t IndexHNSW::insert(const float* q)
{
    size_t id = m_numVectors;

    m_vectors.insert(m_vectors.end(), q, q + m_numDimensions);

    int level = generateLevel();
    m_levels.push_back(level);

    Node newNode;
    newNode.maxLayer = level;
    newNode.neighbors.resize(level + 1);
    
    for (int l = 0; l <= level; ++l)
    {
        size_t Mmax = (l == 0) ? m_Mmax0 : m_Mmax;
        newNode.neighbors[l].reserve(Mmax);
    }

    m_nodes.push_back(std::move(newNode));
    m_nodeLocks.push_back(std::make_shared<std::shared_mutex>());

    insertNode(id);

    m_numVectors++;

    return id;
}

void IndexHNSW::bulkInsert(const float* data, size_t numNewVecs, size_t dims)
{
    if (numNewVecs == 0) return;

    if (m_numVectors == 0)
    {
        m_numDimensions = dims;
    }
    else if (m_numDimensions != dims)
    {
        throw std::runtime_error("IndexHNS::bulkInsert: dimension mismatch");
    }

    size_t oldSize = m_numVectors;
    size_t newTotalSize = oldSize + numNewVecs;

    m_vectors.reserve(newTotalSize * m_numDimensions);
    m_levels.reserve(newTotalSize);
    m_nodes.reserve(newTotalSize);
    m_nodeLocks.reserve(newTotalSize);

    m_vectors.insert(m_vectors.end(), data, data + (numNewVecs * m_numDimensions));

    for (size_t i = 0; i < numNewVecs; i++)
    {
        int lvl = generateLevel();
        m_levels.push_back(lvl);
        
        Node newNode;
        newNode.maxLayer = lvl;
        newNode.neighbors.resize(lvl + 1);
        m_nodes.push_back(std::move(newNode));
        
        m_nodeLocks.push_back(std::make_shared<std::shared_mutex>());
    }

    m_numVectors = newTotalSize;

    if (oldSize == 0)
    {
        insertNode(0);
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 1; i < numNewVecs; i++)
        {
            insertNode(i);
        }
    }
    else
    {
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < numNewVecs; i++)
        {
            insertNode(oldSize + i);
        }
    }
}

std::vector<IndexHNSW::Candidate> IndexHNSW::search(const float* q, size_t k) const
{
    int tempEp = m_entryPoint;
    while (tempEp != -1 && m_deletedVecs.count(tempEp))
    {
        tempEp = -1;
        for (size_t i = 0; i < m_nodes.size(); i++)
        {
            if (!m_deletedVecs.count(i))
            {
                tempEp = i;
                break;
            }
        }
    }

    if (tempEp == -1)
    {
        return {};
    }

    std::vector<Candidate> W;
    std::vector<int> ep = {tempEp};    
    
    for (int lc = m_maxLayer; lc > 0; lc--)
    {
        W = searchLayer(q, ep, 1, lc);
        ep = {(int)W[0].id};
    }

    W = searchLayer(q, ep, m_ef, 0);

    std::vector<Candidate> result;
    for (size_t i = 0; i < k && i < W.size(); i++)
    {
        result.push_back(W[i]);
    }

    return result;
}

void IndexHNSW::save(std::ofstream& f)
{
    // Index Distance Function
    f.write((const char*)&m_metric,         sizeof(m_metric));

    // Vector Size
    f.write((const char*)&m_numVectors,     sizeof(m_numVectors));
    f.write((const char*)&m_numDimensions,  sizeof(m_numDimensions));

    // Graph Construction Parameters
    f.write((const char*)&m_efConstruct,    sizeof(m_efConstruct));
    f.write((const char*)&m_M,              sizeof(m_M));
    f.write((const char*)&m_Mmax,           sizeof(m_Mmax));
    f.write((const char*)&m_Mmax0,          sizeof(m_Mmax0));
    int maxLayer = m_maxLayer.load();
    f.write((const char*)&maxLayer,         sizeof(maxLayer));
    int entryPoint = m_entryPoint.load();
    f.write((const char*)&entryPoint,       sizeof(entryPoint));

    // Flat Vectors
    f.write((const char*)m_vectors.data(),  sizeof(float) * m_numVectors * m_numDimensions);

    // Nodes
    size_t nodesSize = m_nodes.size();
    f.write((const char*)&nodesSize, sizeof(nodesSize));
    for (Node& node : m_nodes)
    {
        f.write((const char*)&node.maxLayer, sizeof(node.maxLayer));

        size_t neighborsSize = node.neighbors.size();
        f.write((const char*)&neighborsSize, sizeof(neighborsSize));
        for (std::vector<int>& neighbor : node.neighbors)
        {
            size_t neighborSize = neighbor.size();
            f.write((const char*)&neighborSize,   sizeof(neighborSize));
            f.write((const char*)neighbor.data(), sizeof(int) * neighborSize);
        }
    }

    // Soft Deleted Vectors
    size_t deletedVecsSize = m_deletedVecs.size();
    f.write((const char*)& deletedVecsSize, sizeof(deletedVecsSize));
    for (auto& id : m_deletedVecs)
    {
        f.write((const char*)&id,    sizeof(id));
    }
}

void IndexHNSW::open(std::ifstream& f)
{
    // Index Distance Function
    f.read((char*)&m_metric,         sizeof(m_metric));

    // Vector Size
    f.read((char*)&m_numVectors,     sizeof(m_numVectors));
    f.read((char*)&m_numDimensions,  sizeof(m_numDimensions));

    // Graph Construction Parameters
    f.read((char*)&m_efConstruct,    sizeof(m_efConstruct));
    f.read((char*)&m_M,              sizeof(m_M));
    f.read((char*)&m_Mmax,           sizeof(m_Mmax));
    f.read((char*)&m_Mmax0,          sizeof(m_Mmax0));
    int maxLayer;
    f.read((char*)&maxLayer,         sizeof(maxLayer));
    m_maxLayer.store(maxLayer);
    int entryPoint;
    f.read((char*)&entryPoint,       sizeof(entryPoint));
    m_entryPoint.store(entryPoint);

    // Flat Vectors
    m_vectors.resize(m_numVectors * m_numDimensions);
    f.read((char*)m_vectors.data(), m_numVectors * m_numDimensions * sizeof(float));

    // Nodes
    size_t nodesSize;
    f.read((char*)&nodesSize, sizeof(nodesSize));
    
    m_nodes.resize(nodesSize);
    for (size_t i = 0; i < nodesSize; i++)
    {
        f.read((char*)&m_nodes[i].maxLayer, sizeof(m_nodes[i].maxLayer));

        size_t neighborsSize;
        f.read((char*)&neighborsSize, sizeof(neighborsSize));
        
        m_nodes[i].neighbors.resize(neighborsSize);
        for (size_t j = 0; j < neighborsSize; j++)
        {
            size_t neighborSize;
            f.read((char*)&neighborSize, sizeof(neighborSize));
            
            m_nodes[i].neighbors[j].resize(neighborSize);
            f.read((char*)m_nodes[i].neighbors[j].data(), neighborSize * sizeof(int));
        }
    }

    // Soft Deleted Vectors
    m_deletedVecs.clear();
    size_t deletedVecsSize;
    f.read((char*)&deletedVecsSize, sizeof(deletedVecsSize));
    for (size_t i = 0; i < deletedVecsSize; i++)
    {
        size_t id;
        f.read((char*)&id,    sizeof(id));
        m_deletedVecs.insert(id);
    }

    // Mutex array
    m_nodeLocks.resize(nodesSize);
    for (auto& lock : m_nodeLocks)
    {
        lock = std::make_shared<std::shared_mutex>();
    }
}


// PRIVATE
void IndexHNSW::insertNode(size_t id)
{
    const float* q = getVector(id);

    std::vector<Candidate> W;
    std::vector<int> ep = {m_entryPoint};
    int L = m_maxLayer;
    int l = m_levels[id];

    if (m_entryPoint == -1)
    {
        int expected = -1;
        if (m_entryPoint.compare_exchange_strong(expected, (int)id))
        {
            m_maxLayer = l;
            return;
        }
    }

    for (int lc = L; lc > l; lc--)
    {
        W = searchLayer(q, ep, 1, lc);
        ep = {(int)W[0].id};
    }

    for (int lc = std::min(L, l); lc >= 0; lc--)
    {
        W = searchLayer(q, ep, m_efConstruct, lc);
        std::vector<Candidate> neighbors = selectNeighbors(q, W, m_M, lc, false);

        // id -> neighbors (one lock, all at once)
        {
            std::unique_lock<std::shared_mutex> lock(*m_nodeLocks[id]);
            for (Candidate& neighbor : neighbors)
            {
                m_nodes[id].neighbors[lc].push_back(neighbor.id);
            }
        }

        // neighbor -> id (with pruning)
        size_t Mmax = (lc == 0) ? m_Mmax0 : m_Mmax;
        for (Candidate& neighbor : neighbors)
        {
            std::unique_lock<std::shared_mutex> lock(*m_nodeLocks[neighbor.id]);
            auto& nbrs = m_nodes[neighbor.id].neighbors[lc];
            nbrs.push_back(id);

            if (nbrs.size() > Mmax)
            {
                std::vector<Candidate> candidates;
                candidates.reserve(nbrs.size());
                for (int nid : nbrs)
                {
                    candidates.push_back({(size_t)nid, d(getVector(neighbor.id), getVector(nid), m_numDimensions)});
                }

                auto pruned = selectNeighbors(getVector(neighbor.id), candidates, Mmax, lc, false);
                nbrs.clear();
                for (auto& c : pruned)
                {
                    nbrs.push_back(c.id);
                }
            }
        }

        ep.clear();
        for (Candidate& candidate : W)
            ep.emplace_back(candidate.id);
    }

    if (l > L)
    {
        m_entryPoint.store(id);
        m_maxLayer.store(l);
    }
}

std::vector<IndexHNSW::Candidate> IndexHNSW::searchLayer(const float* q, const std::vector<int>& ep, size_t ef, int lc) const
{
    thread_local std::vector<int> visitedVersion;
    thread_local int version = 0;

    if (visitedVersion.size() < m_numVectors)
    {
        visitedVersion.assign(m_numVectors, 0);
    }
    
    version++;

    auto visited = [&](int id) { return visitedVersion[id] == version; };
    auto markVisited = [&](int id) { visitedVersion[id] = version; };

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> C;
    std::priority_queue<Candidate> W;

    for (size_t id : ep)
    {

        if (id >= m_nodes.size())
        {
            return {};
        }

        float dist = d(q, getVector(id), m_numDimensions);

        markVisited(id);
        C.push({id, dist});
        W.push({id, dist});
    }

    while (!C.empty())
    {
        Candidate nearestC  = C.top();
        C.pop();

        Candidate farthestW = W.top();

        if (nearestC.distance > farthestW.distance)
            break;

        std::vector<int> neighborsCopy;
        {
            std::shared_lock<std::shared_mutex> lock(*m_nodeLocks[nearestC.id]);
            if (lc < (int)m_nodes[nearestC.id].neighbors.size())
                neighborsCopy = m_nodes[nearestC.id].neighbors[lc];
        }

        for (size_t neighborId : neighborsCopy)
        {
            if (m_deletedVecs.count(neighborId))
            {
                continue;
            }

            if (!visited(neighborId))
            {
                markVisited(neighborId);
                farthestW = W.top();

                float neighborDist = d(q, getVector(neighborId), m_numDimensions);
                if (neighborDist < farthestW.distance || W.size() < ef)
                {
                    C.push({neighborId, neighborDist});
                    W.push({neighborId, neighborDist});

                    if (W.size() > ef)
                        W.pop();
                }
            }
        }
    }
    
    std::vector<Candidate> topEfCandidates;
    topEfCandidates.reserve(ef);

    while (!W.empty())
    {
        topEfCandidates.emplace_back(W.top());
        W.pop();
    }

    std::reverse(topEfCandidates.begin(), topEfCandidates.end());

    return topEfCandidates;
}

std::vector<IndexHNSW::Candidate> IndexHNSW::selectNeighbors(const float* q, const std::vector<Candidate>& candidates, size_t M, int lc, bool extendCandidates) const
{
    std::vector<Candidate> R;
    std::vector<Candidate> Wd;

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> W(candidates.begin(), candidates.end());

    if (extendCandidates)
    {
        for (auto& c : candidates)
        {
            for (int neighborId : m_nodes[c.id].neighbors[lc])
            {
                W.push({(size_t)neighborId, d(q, getVector(neighborId), m_numDimensions)});
            }
        }
    }

    while (!W.empty() && R.size() < M)
    {
        Candidate e = W.top();
        W.pop();
        
        // is e closer to q than to any already selected neighbor?
        bool closerToQuery = true;
        for (auto& r : R)
        {
            if (d(getVector(e.id), getVector(r.id), m_numDimensions) < e.distance)
            {
                closerToQuery = false;
                break;
            }
        }
        
        if (closerToQuery)
            R.push_back(e);
        else
            Wd.push_back(e);
    }

    while (!Wd.empty() && R.size() < M)
    {
        R.push_back(Wd.front());
        Wd.erase(Wd.begin());
    }

    return R;
}

int IndexHNSW::generateLevel() const
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    return (int)(-std::log(dist(m_rng)) * m_mL);
}

} /* namespace demDB */