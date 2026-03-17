#include "../src/dementiaDB.h"
#include "../utils/mathutils.h"
#include "../utils/timer.h"

#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_set>


int main()
{
    const std::string colName = "books";

    demDB::DementiaDB db;
    
    long long openTimeMs = 0;
    {
        ScopeTimer t(openTimeMs);

        db.open("test_db.dem");
    }

    std::cout << "Opened in: " << openTimeMs << std::endl;


    return 0;
}