#include <fakeit.hpp>
#include <trompeloeil.hpp>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

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
