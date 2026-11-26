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
  int32_t type = PrivateUse1HooksInterface::getDLDeviceType();

  DeviceIndex getCurrentDevice() const override {
    return 0;
  }

  int32_t getDLDeviceType() const override {
    return type;
  }
};
} // namespace

TEST(TestDlconvertorDeathTest, TestPrivateUse1DeviceHook) {
  // Hook registration is permanent, so isolate it from other tests.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        const Device device(kPrivateUse1, 0);
        auto dl_device = torchDeviceToDLDevice(device);
        EXPECT_EQ(dl_device.device_type, kDLExtDev);
        EXPECT_EQ(dl_device.device_id, 0);

        TestPrivateUse1Hooks hooks;
        RegisterPrivateUse1HooksInterface(&hooks);
        dl_device = torchDeviceToDLDevice(device);
        EXPECT_EQ(dl_device.device_type, kDLExtDev);
        EXPECT_EQ(dl_device.device_id, 0);

        hooks.type = kDLCUDAManaged;
        dl_device = torchDeviceToDLDevice(Device(kPrivateUse1));
        EXPECT_EQ(dl_device.device_type, kDLCUDAManaged);
        EXPECT_EQ(dl_device.device_id, 0);
        EXPECT_EQ(dlDeviceToTorchDevice(kDLCUDAManaged, 0), device);

        // Host storage suffices to test metadata and aliasing without kernels.
        float data[6] = {};
        auto tensor = for_blob(data, {6})
                          .options(TensorOptions().device(device).dtype(kFloat))
                          .target_device(device)
                          .make_tensor();
        auto imported = fromDLPack(toDLPack(tensor));
        EXPECT_EQ(imported.device(), device);
        EXPECT_EQ(imported.data_ptr(), tensor.data_ptr());
        EXPECT_EQ(imported.sizes(), tensor.sizes());
        EXPECT_TRUE(maybeCopyTensor(tensor, dl_device, false).is_same(tensor));

        hooks.type = kDLCPU;
        EXPECT_THROW(torchDeviceToDLDevice(device), c10::Error);

        std::_Exit(::testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS),
      "");
}
#endif // GTEST_HAS_DEATH_TEST
