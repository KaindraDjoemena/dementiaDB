#include <chrono>


class ScopeTimer {
public:
    ScopeTimer(long long& accumulator)
        : m_accum(accumulator)
        , start(std::chrono::high_resolution_clock::now()) {}

    ~ScopeTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        m_accum += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

private:
    long long& m_accum;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};