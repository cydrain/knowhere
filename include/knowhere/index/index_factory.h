// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef INDEX_FACTORY_H
#define INDEX_FACTORY_H

#include <functional>
#include <set>
#include <string>
#include <unordered_map>

#include "index_static.h"
#include "knowhere/index/index.h"
#include "knowhere/utils.h"
#include "simd/hook.h"
#ifdef KNOWHERE_WITH_RAFT
#include <cuda_runtime_api.h>
#endif

namespace knowhere {
#ifdef KNOWHERE_WITH_RAFT
inline bool
checkGpuAvailable(const std::string& name) {
    if (name == "GPU_RAFT_BRUTE_FORCE" || name == "GPU_BRUTE_FORCE" || name == "GPU_RAFT_CAGRA" ||
        name == "GPU_CAGRA" || name == "GPU_RAFT_IVF_FLAT" || name == "GPU_IVF_FLAT" || name == "GPU_RAFT_IVF_PQ" ||
        name == "GPU_IVF_PQ") {
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess) {
            LOG_KNOWHERE_INFO_ << cudaGetErrorString(status);
            return false;
        }
        if (count < 1) {
            LOG_KNOWHERE_INFO_ << "GPU not available";
            return false;
        }
    }
    return true;
}
#endif

class IndexFactory {
 public:
    template <typename DataType>
    expected<Index<IndexNode>>
    Create(const std::string& name, const int32_t& version, const Object& object = nullptr) {
        static_assert(KnowhereDataTypeCheck<DataType>::value == true);
        auto& func_mapping_ = MapInstance();
        auto key = GetKey<DataType>(name);
        if (func_mapping_.find(key) == func_mapping_.end()) {
            LOG_KNOWHERE_ERROR_ << "failed to find index " << key << " in factory";
            return expected<Index<IndexNode>>::Err(Status::invalid_index_error, "index not supported");
        }
        LOG_KNOWHERE_INFO_ << "use key " << key << " to create knowhere index " << name << " with version " << version;
        auto fun_map_v = (FunMapValue<Index<IndexNode>>*)(func_mapping_[key].get());

#ifdef KNOWHERE_WITH_RAFT
        if (!checkGpuAvailable(name)) {
            return expected<Index<IndexNode>>::Err(Status::cuda_runtime_error, "gpu not available");
        }
#endif
        if (name == knowhere::IndexEnum::INDEX_FAISS_SCANN && !faiss::support_pq_fast_scan) {
            LOG_KNOWHERE_ERROR_ << "SCANN index is not supported on the current CPU model";
            return expected<Index<IndexNode>>::Err(Status::invalid_index_error,
                                                   "SCANN index is not supported on the current CPU model");
        }

        return fun_map_v->fun_value(version, object);
    }

    template <typename DataType>
    const IndexFactory&
    Register(const std::string& name, std::function<Index<IndexNode>(const int32_t&, const Object&)> func) {
        static_assert(KnowhereDataTypeCheck<DataType>::value == true);
        auto& func_mapping_ = MapInstance();
        auto key = GetKey<DataType>(name);
        assert(func_mapping_.find(key) == func_mapping_.end());
        func_mapping_[key] = std::make_unique<FunMapValue<Index<IndexNode>>>(func);
        return *this;
    }

    static IndexFactory&
    Instance() {
        static IndexFactory factory;
        return factory;
    }

    typedef std::tuple<std::set<std::pair<std::string, VecType>>, std::set<std::string>> GlobalIndexTable;
    static GlobalIndexTable&
    StaticIndexTableInstance() {
        static GlobalIndexTable static_index_table;
        return static_index_table;
    }

 private:
    struct FunMapValueBase {
        virtual ~FunMapValueBase() = default;
    };

    template <typename T1>
    struct FunMapValue : FunMapValueBase {
     public:
        FunMapValue(std::function<T1(const int32_t&, const Object&)>& input) : fun_value(input) {
        }
        std::function<T1(const int32_t&, const Object&)> fun_value;
    };

    IndexFactory() = default;

    typedef std::map<std::string, std::unique_ptr<FunMapValueBase>> FuncMap;
    static FuncMap&
    MapInstance() {
        static FuncMap func_map;
        return func_map;
    }
};

#define KNOWHERE_FACTORY_CONCAT(x, y) index_factory_ref_##x##y
#define KNOWHERE_STATIC_CONCAT(x, y) index_static_ref_##x##y

#define KNOWHERE_REGISTER_STATIC(name, index_node, data_type, ...)               \
    const IndexStaticFaced<data_type>& KNOWHERE_STATIC_CONCAT(name, data_type) = \
        IndexStaticFaced<data_type>::Instance().RegisterStaticFunc<index_node<data_type, ##__VA_ARGS__>>(#name);

#define KNOWHERE_REGISTER_GLOBAL(name, func, data_type)            \
    const IndexFactory& KNOWHERE_FACTORY_CONCAT(name, data_type) = \
        IndexFactory::Instance().Register<data_type>(#name, func)

#define KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, data_type, ...)                             \
    KNOWHERE_REGISTER_STATIC(name, index_node, data_type, ##__VA_ARGS__)                              \
    KNOWHERE_REGISTER_GLOBAL(                                                                         \
        name,                                                                                         \
        (static_cast<Index<index_node<data_type, ##__VA_ARGS__>> (*)(const int32_t&, const Object&)>( \
            &Index<index_node<data_type, ##__VA_ARGS__>>::Create)),                                   \
        data_type)
#define KNOWHERE_MOCK_REGISTER_GLOBAL(name, index_node, data_type, ...)                                    \
    KNOWHERE_REGISTER_STATIC(name, index_node, data_type, ##__VA_ARGS__)                                   \
    KNOWHERE_REGISTER_GLOBAL(                                                                              \
        name,                                                                                              \
        [](const int32_t& version, const Object& object) {                                                 \
            return (Index<IndexNodeDataMockWrapper<data_type>>::Create(                                    \
                std::make_unique<index_node<MockData<data_type>::type, ##__VA_ARGS__>>(version, object))); \
        },                                                                                                 \
        data_type)
#define KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(name, index_node, data_type, thread_size)              \
    KNOWHERE_REGISTER_STATIC(name, index_node, data_type)                                                \
    KNOWHERE_REGISTER_GLOBAL(                                                                            \
        name,                                                                                            \
        [](const int32_t& version, const Object& object) {                                               \
            return (Index<IndexNodeThreadPoolWrapper>::Create(                                           \
                std::make_unique<index_node<MockData<data_type>::type>>(version, object), thread_size)); \
        },                                                                                               \
        data_type)
#define KNOWHERE_SET_STATIC_GLOBAL_INDEX_TABLE(table_index, name, index_table)                      \
    static int name = []() -> int {                                                                 \
        auto& static_index_table = std::get<table_index>(IndexFactory::StaticIndexTableInstance()); \
        static_index_table.insert(index_table.begin(), index_table.end());                          \
        return 0;                                                                                   \
    }();
}  // namespace knowhere

#endif /* INDEX_FACTORY_H */
