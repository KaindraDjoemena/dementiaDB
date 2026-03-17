#include "../src/indexHnsw.h"
#include "../utils/timer.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <unordered_set>
#include <highfive/highfive.hpp>
#include <omp.h>

using namespace demDB;

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_hdf5.hdf5> [metric] [k_search]" << std::endl;
        std::cerr << "  metric: l2 (default), cos, ncos, ip" << std::endl;
        std::cerr << "  k_search: number of neighbors to retrieve (default: 10)" << std::endl;
        return 1;
    }

    std::filesystem::path hdf5Path(argv[1]);
    if (!std::filesystem::exists(hdf5Path))
        throw std::runtime_error("Invalid path " + hdf5Path.string());

    size_t k_search        = 10;
    METRIC metric          = METRIC::L2;
    std::string metricName = "l2";

    if (argc >= 3)
    {
        std::string metricStr = argv[2];
        if      (metricStr == "l2")   { metric = METRIC::L2;         metricName = "l2";   }
        else if (metricStr == "cos")  { metric = METRIC::COSINE;     metricName = "cos";  }
        else if (metricStr == "ncos") { metric = METRIC::NORM_COSINE; metricName = "ncos"; }
        else if (metricStr == "ip")   { metric = METRIC::INNER_PROD; metricName = "ip";   }
        else throw std::invalid_argument("Invalid metric " + metricStr);
    }

    if (argc >= 4)
        k_search = std::stoul(argv[3]);

    size_t M               = 16;
    size_t M_max0          = 32;
    size_t ef_construction = 100;

    demDB::IndexHNSW index;
    index.setMetric(metric);
    index.setM(M);
    index.setMmax(M);
    index.setMmax0(M_max0);
    index.setEfConstruction(ef_construction);

    HighFive::File file(hdf5Path, HighFive::File::ReadOnly);

    HighFive::DataSet trainDS = file.getDataSet("train");
    auto trainDims  = trainDS.getDimensions();
    size_t numTrain = trainDims[0];
    size_t dims     = trainDims[1];
    std::vector<float> trainData(numTrain * dims);
    trainDS.read(trainData.data());

    HighFive::DataSet queryDS = file.getDataSet("test");
    auto queryDims    = queryDS.getDimensions();
    size_t numQueries = queryDims[0];
    std::vector<float> queryData(numQueries * dims);
    queryDS.read(queryData.data());

    HighFive::DataSet gtDS = file.getDataSet("neighbors");
    auto gtDims = gtDS.getDimensions();
    size_t k_gt = gtDims[1];
    std::vector<int> gtData(numQueries * k_gt);
    gtDS.read(gtData.data());

    if (k_search > k_gt)
    {
        std::cerr << "Warning: k_search (" << k_search << ") > ground truth size (" << k_gt << "). Clamping to " << k_gt << std::endl;
        k_search = k_gt;
    }

    long long buildTime = 0;
    {
        ScopeTimer t(buildTime);
        index.bulkInsert(trainData.data(), numTrain, dims);
    }

    std::vector<size_t> ef_vals = {10, 50, 100, 200, 400, 800};

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "{\n";
    std::cout << "  \"implementation\": \"dementiaDB\",\n";
    std::cout << "  \"dataset\": \""        << hdf5Path.stem().string() << "\",\n";
    std::cout << "  \"metric\": \""         << metricName      << "\",\n";
    std::cout << "  \"M\": "                << M               << ",\n";
    std::cout << "  \"ef_construction\": "  << ef_construction << ",\n";
    std::cout << "  \"build_time_ms\": "    << buildTime       << ",\n";
    std::cout << "  \"results\": [\n";

    for (size_t efIdx = 0; efIdx < ef_vals.size(); ++efIdx)
    {
        size_t ef = ef_vals[efIdx];
        index.setEf(ef);

        float     totalRecall = 0;
        long long searchTime  = 0;
        {
            ScopeTimer t(searchTime);
            #pragma omp parallel for reduction(+:totalRecall) schedule(dynamic)
            for (size_t i = 0; i < numQueries; i++)
            {
                auto results = index.search(queryData.data() + i * dims, k_search);

                std::unordered_set<int> gt;
                for (size_t j = 0; j < k_search; ++j)
                    gt.insert(gtData[i * k_gt + j]);

                size_t hits = 0;
                for (auto& res : results)
                    if (gt.count(res.id)) hits++;

                totalRecall += (float)hits / k_search;
            }
        }

        float recall  = totalRecall / numQueries;
        float qps     = (float)numQueries / (searchTime / 1000.0f);
        float latency = (float)searchTime / numQueries;

        std::cout << "    {\n";
        std::cout << "      \"ef\": "         << ef       << ",\n";
        std::cout << "      \"recall\": "     << recall   << ",\n";
        std::cout << "      \"qps\": "        << (int)qps << ",\n";
        std::cout << "      \"latency_ms\": " << latency  << "\n";
        std::cout << "    }" << (efIdx + 1 < ef_vals.size() ? "," : "") << "\n";
    }

    std::cout << "  ]\n}\n";

    return 0;
}