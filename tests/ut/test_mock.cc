#include <fakeit.hpp>
#include <trompeloeil.hpp>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "mock/index_factory_mock.h"
#include "utils.h"

class Warehouse {
 public:
    virtual bool
    hasInventory(const std::string& name, size_t amount) const = 0;
    virtual void
    remove(const std::string& name, size_t amount) = 0;
};

class Order {
 public:
    Order(const std::string& name, size_t amount){};
    void
    fill(Warehouse& w){};
    bool
    is_filled() const {
        return true;
    };
};

class WarehouseMock : public Warehouse {
 public:
    MAKE_CONST_MOCK2(hasInventory, bool(const std::string&, size_t));
    MAKE_MOCK2(remove, void(const std::string&, size_t));
};

TEST_CASE("test trompeloeil") {
    using trompeloeil::_;
    Order order("Talisker", 50);

    WarehouseMock warehouse;
    {
        REQUIRE_CALL(warehouse, hasInventory("Talisker", 50)).RETURN(true);
        REQUIRE_CALL(warehouse, remove("Talisker", 50));
        order.fill(warehouse);
    }

    REQUIRE(order.is_filled());
}

class IFoo {
 public:
    virtual ~IFoo() = default;
    virtual int
    foo(int x) const {
        return x;
    };
};

TEST_CASE("Test FakeIt") {
    using namespace fakeit;
    Mock<IFoo> mock_foo;

    SECTION("Mocking a method call") {
        When(Method(mock_foo, foo)).Return(234);
        auto x = mock_foo.get().foo(123);
        REQUIRE(x == 456);
    }
}

TEST_CASE("Test Mem Index With Float Vector", "[float metrics]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);
    auto topk = GENERATE(as<int64_t>{}, 5, 120);
    auto version = GenTestVersionList();

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto ivfflat_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 8;
        return json;
    };

    auto flat_gen = base_gen;

    auto hnsw_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 32;
        json[knowhere::indexparam::EFCONSTRUCTION] = 120;
        json[knowhere::indexparam::EF] = 120;
        return json;
    };

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

    SECTION("Test Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        knowhere::BinarySet bs;
        // build process
        {
            auto idx_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
            auto idx = idx_expected.value();
            auto cfg_json = gen().dump();
            CAPTURE(name, cfg_json);
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(idx.Type() == name);
            REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
            REQUIRE(idx.Size() > 0);
            REQUIRE(idx.Count() == nb);

            REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
        }
    }
}
