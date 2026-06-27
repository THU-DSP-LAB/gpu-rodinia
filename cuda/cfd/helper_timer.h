#pragma once

#include <chrono>

struct StopWatchInterface {
    std::chrono::steady_clock::time_point start;
    double elapsed_ms = 0.0;
};

inline void sdkCreateTimer(StopWatchInterface** timer) {
    *timer = new StopWatchInterface();
}

inline void sdkStartTimer(StopWatchInterface** timer) {
    (*timer)->start = std::chrono::steady_clock::now();
}

inline void sdkStopTimer(StopWatchInterface** timer) {
    const auto end = std::chrono::steady_clock::now();
    (*timer)->elapsed_ms += std::chrono::duration<double, std::milli>(end - (*timer)->start).count();
}

inline double sdkGetAverageTimerValue(StopWatchInterface** timer) {
    return (*timer)->elapsed_ms;
}
