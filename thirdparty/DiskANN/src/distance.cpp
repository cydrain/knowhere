#include <cmath>
#include "diskann/distance.h"
namespace diskann {
  // Get the right distance function for the given metric.
  template<typename T>
  DISTFUN<T> get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      return [](const T* x, const T* y, size_t size) -> float {
        float res = 0;
        for (size_t i = 0; i < size; i++) {
          res += ((float) x[i] - (float) y[i]) * ((float) x[i] - (float) y[i]);
        }
        return res;
      };
    } else if (m == diskann::Metric::INNER_PRODUCT ||
               m == diskann::Metric::COSINE) {
      return [](const T* x, const T* y, size_t size) -> float {
        float res = 0;
        for (size_t i = 0; i < size; i++) {
          res += (float) x[i] * (float) y[i];
        }
        return -res;
      };
    } else {
      std::stringstream stream;
      stream << "Only L2 and inner product supported as for now. ";
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  DISTFUN<float> get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      return faiss::fvec_L2sqr;
    } else if (m == diskann::Metric::INNER_PRODUCT) {
      return [](const float* x, const float* y, size_t size) -> float {
        return (-1.0) * faiss::fvec_inner_product(x, y, size);
      };
    } else if (m == diskann::Metric::COSINE) {
      return [](const float* x, const float* y, size_t size) -> float {
        return (-1.0) * faiss::fvec_inner_product(x, y, size);
      };
    } else {
      std::stringstream stream;
      stream << "Only L2, cosine, and inner product supported for floating "
                "point vectors as of now. ";
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  DISTFUN<knowhere::fp16> get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      return faiss::fp16_vec_L2sqr;
    } else if (m == diskann::Metric::INNER_PRODUCT) {
      return [](const knowhere::fp16* x, const knowhere::fp16* y,
                size_t size) -> float {
        return (-1.0) * faiss::fp16_vec_inner_product(x, y, size);
      };
    } else if (m == diskann::Metric::COSINE) {
      return [](const knowhere::fp16* x, const knowhere::fp16* y,
                size_t size) -> float {
        return (-1.0) * faiss::fp16_vec_inner_product(x, y, size);
      };
    } else {
      std::stringstream stream;
      stream << "Only L2, cosine, and inner product supported for float16 "
                "vectors as of now. ";
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  DISTFUN<knowhere::bf16> get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      return faiss::bf16_vec_L2sqr;
    } else if (m == diskann::Metric::INNER_PRODUCT) {
      return [](const knowhere::bf16* x, const knowhere::bf16* y,
                size_t size) -> float {
        return (-1.0) * faiss::bf16_vec_inner_product(x, y, size);
      };
    } else if (m == diskann::Metric::COSINE) {
      return [](const knowhere::bf16* x, const knowhere::bf16* y,
                size_t size) -> float {
        return (-1.0) * faiss::bf16_vec_inner_product(x, y, size);
      };
    } else {
      std::stringstream stream;
      stream << "Only L2, cosine, and inner product supported for bfloat16"
                "vectors as of now. ";
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  // get vector sqr norm
  template<typename T>
  float norm_l2sqr(const T* a, size_t size) {
    if constexpr (std::is_floating_point<T>::value) {
      return faiss::fvec_norm_L2sqr(a, size);
    }
    if constexpr (std::is_same_v<T, knowhere::fp16>) {
      return faiss::fp16_vec_norm_L2sqr(a, size);
    }
    if constexpr (std::is_same_v<T, knowhere::bf16>) {
      return faiss::bf16_vec_norm_L2sqr(a, size);
    } else {
      float res = 0;
      for (size_t i = 0; i < size; i++) {
        res += (float) a[i] * (float) a[i];
      }
      return res;
    }
  }

  template DISTFUN<float>          get_distance_function(diskann::Metric m);
  template DISTFUN<uint8_t>        get_distance_function(diskann::Metric m);
  template DISTFUN<int8_t>         get_distance_function(diskann::Metric m);
  template DISTFUN<knowhere::fp16> get_distance_function(diskann::Metric m);
  template DISTFUN<knowhere::bf16> get_distance_function(diskann::Metric m);

  template float norm_l2sqr(const float*, size_t);
  template float norm_l2sqr(const uint8_t*, size_t);
  template float norm_l2sqr(const int8_t*, size_t);
  template float norm_l2sqr(const knowhere::fp16*, size_t);
  template float norm_l2sqr(const knowhere::bf16*, size_t);
}  // namespace diskann
