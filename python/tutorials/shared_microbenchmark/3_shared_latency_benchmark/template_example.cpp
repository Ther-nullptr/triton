#include <iostream>
#include <tuple>

struct Param {
    int value;
    // 可以在这里添加 Param 结构体的其他成员
};

template <int Warps, int Threads>
struct SharedLatency {
    static void callKernel(const Param& param) {
        std::cout << "shared_lat<" << Warps << ", " << Threads << ">\n";
        // 在这里添加对应的 CUDA kernel 调用，可以使用 param.value
        std::cout << "Param Value: " << param.value << std::endl;
    }
};

template <typename... Functions, std::size_t... Is>
void callAllFunctionsImpl(const std::tuple<Functions...>& functionList, const Param& param, std::index_sequence<Is...>) {
    ((std::get<Is>(functionList))(param), ...);
}

template <typename... Functions>
void callAllFunctions(const std::tuple<Functions...>& functionList, const Param& param) {
    callAllFunctionsImpl(functionList, param, std::index_sequence_for<Functions...>{});
}

template <int WarpsStart, int WarpsEnd, int ThreadsStart, int ThreadsEnd>
struct GenerateSharedLat {
    static auto generate() {
        if constexpr (ThreadsStart <= ThreadsEnd) {
            return std::tuple_cat(
                std::tuple<void (*)(const Param&)>{&SharedLatency<WarpsStart, ThreadsStart>::callKernel},
                GenerateSharedLat<WarpsStart, WarpsEnd, ThreadsStart * 2, ThreadsEnd>::generate());
        } else if constexpr (WarpsStart <= WarpsEnd) {
            return std::tuple_cat(
                std::tuple<void (*)(const Param&)>{&SharedLatency<WarpsStart, ThreadsStart>::callKernel},
                GenerateSharedLat<WarpsStart * 2, WarpsEnd, 1, ThreadsEnd>::generate());
        } else {
            return std::tuple<>();
        }
    }
};

int main() {
    constexpr int WarpsStart = 1;
    constexpr int WarpsEnd = 32;
    constexpr int ThreadsStart = 1;
    constexpr int ThreadsEnd = 32;
    Param param{42};

    auto functionList = GenerateSharedLat<WarpsStart, WarpsEnd, ThreadsStart, ThreadsEnd>::generate();
    callAllFunctions(functionList, param);

    return 0;
}