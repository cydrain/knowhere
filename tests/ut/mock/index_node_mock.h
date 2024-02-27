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

#include <trompeloeil.hpp>

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index_node.h"

namespace knowhere {
class IndexNodeMock : public knowhere::IndexNode {
 public:
    MAKE_MOCK2(Build, Status(const DataSet&, const Config&));
    MAKE_MOCK2(Train, Status(const DataSet&, const Config&));
    MAKE_MOCK2(Add, Status(const DataSet&, const Config&));

    MAKE_CONST_MOCK3(Search, expected<DataSetPtr>(const DataSet&, const Config&, const BitsetView&));
    MAKE_CONST_MOCK3(AnnIterator, expected<std::vector<std::shared_ptr<iterator>>>(const DataSet&, const Config&,
                                                                                   const BitsetView&));
    MAKE_CONST_MOCK3(RangeSearch, expected<DataSetPtr>(const DataSet&, const Config&, const BitsetView&));

    MAKE_CONST_MOCK1(GetVectorByIds, expected<DataSetPtr>(const DataSet&));
    MAKE_CONST_MOCK1(HasRawData, bool(const std::string&));
    MAKE_CONST_MOCK0(IsAdditionalScalarSupported, bool());
    MAKE_CONST_MOCK1(GetIndexMeta, expected<DataSetPtr>(const Config&));
    MAKE_CONST_MOCK1(Serialize, Status(BinarySet&));
    MAKE_MOCK2(Deserialize, Status(const BinarySet&, const Config&));
    MAKE_MOCK2(DeserializeFromFile, Status(const std::string&, const Config&));
    MAKE_CONST_MOCK0(CreateConfig, std::unique_ptr<BaseConfig>());

    MAKE_CONST_MOCK0(Dim, int64_t());
    MAKE_CONST_MOCK0(Size, int64_t());
    MAKE_CONST_MOCK0(Count, int64_t());
    MAKE_CONST_MOCK0(Type, std::string());
};

}  // namespace knowhere
