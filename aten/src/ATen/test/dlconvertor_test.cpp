#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <cstdlib>

using namespace at;

TEST(TestDlconvertor, TestDlconvertor) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensor* dlMTensor = toDLPack(a);

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorNoStrides) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensor* dlMTensor = toDLPack(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorUnversioned, TestDlconvertor) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorUnversioned, TestDlconvertorNoStrides) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

#if GTEST_HAS_DEATH_TEST
namespace {
struct TestPrivateUse1Hooks : PrivateUse1HooksInterface {
  DLDevice getDLDevice(DeviceIndex device_index) const override {
    EXPECT_EQ(device_index, 1);
    return {DLDeviceType::kDLCPU, 0};
  }
};
} // namespace

TEST(TestDlconvertorDeathTest, TestPrivateUse1DeviceHook) {
  // Hook registration is permanent, so isolate it from other tests.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        const Device device(kPrivateUse1, 1);
        DLDevice dl_device = torchDeviceToDLDevice(device);
        EXPECT_EQ(dl_device.device_type, DLDeviceType::kDLExtDev);
        EXPECT_EQ(dl_device.device_id, 1);

        TestPrivateUse1Hooks hooks;
        RegisterPrivateUse1HooksInterface(&hooks);
        dl_device = torchDeviceToDLDevice(device);
        EXPECT_EQ(dl_device.device_type, DLDeviceType::kDLCPU);
        EXPECT_EQ(dl_device.device_id, 0);

        std::_Exit(::testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS),
      "");
}
#endif // GTEST_HAS_DEATH_TEST
