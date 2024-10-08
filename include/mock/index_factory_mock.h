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

#pragma once

#include <functional>
#include <set>
#include <string>
#include <unordered_map>

#include "knowhere/index/index.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "simd/hook.h"

namespace knowhere {

class IndexFactoryMock {
 public:
    template <typename DataType>
    expected<Index<IndexNode>>
    Create(const std::string& name, const int32_t& version, const Object& object = nullptr) {
        static_assert(KnowhereDataTypeCheck<DataType>::value == true);
        auto& func_mapping_ = MapInstance();
        auto key = GetKey<DataType>(name);
        if (func_mapping_.find(key) == func_mapping_.end()) {
            LOG_KNOWHERE_ERROR_ << "failed to find index " << key << " in factory mock";
            return expected<Index<IndexNode>>::Err(Status::invalid_index_error, "index not supported");
        }
        LOG_KNOWHERE_INFO_ << "use key " << key << " to create knowhere index " << name << " with version " << version;
        auto fun_map_v = (FunMapValue<Index<IndexNode>>*)(func_mapping_[key].get());

        return fun_map_v->fun_value(version, object);
    }

    template <typename DataType>
    const IndexFactoryMock&
    Register(const std::string& name, std::function<Index<IndexNode>(const int32_t&, const Object&)> func) {
        static_assert(KnowhereDataTypeCheck<DataType>::value == true);
        auto& func_mapping_ = MapInstance();
        auto key = GetKey<DataType>(name);
        assert(func_mapping_.find(key) == func_mapping_.end());
        func_mapping_[key] = std::make_unique<FunMapValue<Index<IndexNode>>>(func);
        return *this;
    }

    static IndexFactoryMock&
    Instance() {
        static IndexFactoryMock factory;
        return factory;
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

    IndexFactoryMock() = default;

    typedef std::map<std::string, std::unique_ptr<FunMapValueBase>> FuncMap;
    static FuncMap&
    MapInstance() {
        static FuncMap func_map;
        return func_map;
    }
};

#define KNOWHERE_FACTORY_MOCK_CONCAT(x, y) index_factory_mock_ref_##x##y

#define KNOWHERE_REGISTER_GLOBAL_MOCK(name, func, data_type)                \
    const IndexFactoryMock& KNOWHERE_FACTORY_MOCK_CONCAT(name, data_type) = \
        IndexFactoryMock::Instance().Register<data_type>(#name, func)

#define KNOWHERE_SIMPLE_REGISTER_GLOBAL_MOCK(name, index_node, data_type, ...)                        \
    KNOWHERE_REGISTER_GLOBAL_MOCK(                                                                    \
        name,                                                                                         \
        (static_cast<Index<index_node<data_type, ##__VA_ARGS__>> (*)(const int32_t&, const Object&)>( \
            &Index<index_node<data_type, ##__VA_ARGS__>>::Create)),                                   \
        data_type)
}  // namespace knowhere
