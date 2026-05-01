/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Definition of experimental bucketed tail layer for NNUE evaluation function

#ifndef NNUE_LAYERS_BUCKETED_TAIL_H_INCLUDED
#define NNUE_LAYERS_BUCKETED_TAIL_H_INCLUDED

#include "../nnue_common.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

namespace Eval::NNUE::Layers {

  // Experimental bucketed tail:
  //
  //   PreviousLayer output
  //     -> bucketed affine 32 -> HiddenDimensions
  //     -> clipped ReLU
  //     -> bucketed affine HiddenDimensions -> 1
  //
  // For the first scaffold version, bucket selection is fixed to bucket 0.
  template <typename PreviousLayer, IndexType BucketCount, IndexType HiddenDimensions>
  class BucketedTail {
   public:
    // Input/output type
    using InputType = typename PreviousLayer::OutputType;
    using OutputType = std::int32_t;
    static_assert(std::is_same<InputType, std::uint8_t>::value, "");

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions =
        PreviousLayer::kOutputDimensions;
    static constexpr IndexType kHiddenDimensions = HiddenDimensions;
    static constexpr IndexType kOutputDimensions = 1;
    static constexpr IndexType kBucketCount = BucketCount;

    static_assert(kBucketCount > 0, "");
    static_assert(kHiddenDimensions > 0, "");

    static constexpr IndexType kPaddedInputDimensions =
        CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);
    static constexpr IndexType kPaddedHiddenDimensions =
        CeilToMultiple<IndexType>(kHiddenDimensions, kMaxSimdWidth);

    // Forward buffer:
    //   output[1] first, because Propagate() returns this pointer
    //   hidden clipped activations after that
    static constexpr std::size_t kOutputBufferSize =
        CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);
    static constexpr std::size_t kHiddenBufferSize =
        CeilToMultiple(kHiddenDimensions * sizeof(std::uint8_t), kCacheLineSize);

    static constexpr std::size_t kSelfBufferSize =
        kOutputBufferSize + kHiddenBufferSize;

    static constexpr std::size_t kBufferSize =
        PreviousLayer::kBufferSize + kSelfBufferSize;

    static constexpr int kLayerIndex = PreviousLayer::kLayerIndex + 1;

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {
      std::uint32_t hash_value = 0xB42D6A3Fu;
      hash_value += kBucketCount;
      hash_value ^= kHiddenDimensions << 8;
      hash_value ^= PreviousLayer::GetHashValue() >> 1;
      hash_value ^= PreviousLayer::GetHashValue() << 31;
      return hash_value;
    }

    static std::string get_name() {
      return "BucketedTail[" +
          std::to_string(kBucketCount) + "x(" +
          std::to_string(kInputDimensions) + "->" +
          std::to_string(kHiddenDimensions) + "->1)]";
    }

    // A string that represents the structure from the input layer to this layer
    static std::string get_structure_string() {
      return get_name() + "(" +
          PreviousLayer::get_structure_string() + ")";
    }

    static std::string get_layers_info() {
      std::string info = PreviousLayer::get_layers_info();
      info += "\n  - ";
      info += std::to_string(kLayerIndex);
      info += " - ";
      info += get_name();
      return info;
    }

    // Read network parameters
    bool ReadParameters(std::istream& stream) {
      if (!previous_layer_.ReadParameters(stream))
        return false;

      for (std::size_t i = 0; i < kBucketCount * kHiddenDimensions; ++i)
        hidden_biases_[i] = read_little_endian<BiasType>(stream);

      for (std::size_t i = 0;
           i < kBucketCount * kHiddenDimensions * kPaddedInputDimensions;
           ++i)
        hidden_weights_[i] = read_little_endian<WeightType>(stream);

      for (std::size_t i = 0; i < kBucketCount; ++i)
        output_biases_[i] = read_little_endian<BiasType>(stream);

      for (std::size_t i = 0; i < kBucketCount * kPaddedHiddenDimensions; ++i)
        output_weights_[i] = read_little_endian<WeightType>(stream);

      return !stream.fail();
    }

    // Write parameters
    bool WriteParameters(std::ostream& stream) const {
      if (!previous_layer_.WriteParameters(stream))
        return false;

      stream.write(reinterpret_cast<const char*>(hidden_biases_),
          kBucketCount * kHiddenDimensions * sizeof(BiasType));

      stream.write(reinterpret_cast<const char*>(hidden_weights_),
          kBucketCount * kHiddenDimensions * kPaddedInputDimensions *
          sizeof(WeightType));

      stream.write(reinterpret_cast<const char*>(output_biases_),
          kBucketCount * sizeof(BiasType));

      stream.write(reinterpret_cast<const char*>(output_weights_),
          kBucketCount * kPaddedHiddenDimensions * sizeof(WeightType));

      return !stream.fail();
    }

    // Forward propagation
    const OutputType* Propagate(
        const TransformedFeatureType* transformed_features, char* buffer) const {

      const auto input = previous_layer_.Propagate(
          transformed_features, buffer + kSelfBufferSize);

      auto output = reinterpret_cast<OutputType*>(buffer);
      auto hidden = reinterpret_cast<std::uint8_t*>(buffer + kOutputBufferSize);

      const IndexType bucket = GetBucketIndex(transformed_features);

      const auto hidden_bias_base =
          bucket * kHiddenDimensions;
      const auto hidden_weight_base =
          bucket * kHiddenDimensions * kPaddedInputDimensions;

      for (IndexType i = 0; i < kHiddenDimensions; ++i) {
        OutputType sum = hidden_biases_[hidden_bias_base + i];

        const auto weight_offset =
            hidden_weight_base + i * kPaddedInputDimensions;

        for (IndexType j = 0; j < kInputDimensions; ++j)
          sum += hidden_weights_[weight_offset + j] * input[j];

        hidden[i] = static_cast<std::uint8_t>(
            std::max(0, std::min(127, sum >> kWeightScaleBits)));
      }

      OutputType sum = output_biases_[bucket];

      const auto output_weight_base =
          bucket * kPaddedHiddenDimensions;

      for (IndexType j = 0; j < kHiddenDimensions; ++j)
        sum += output_weights_[output_weight_base + j] * hidden[j];

      output[0] = sum;
      return output;
    }

   private:
    using BiasType = OutputType;
    using WeightType = std::int8_t;

    // Make the learning class a friend
    friend class Trainer<BucketedTail>;

    // TODO:
    // The current layer Propagate() API does not receive Position.
    // This scaffold therefore always selects bucket 0.
    // Real phase / piece-count bucketing needs a proper bucket source.
    static IndexType GetBucketIndex(
        const TransformedFeatureType* /*transformed_features*/) {
      return 0;
    }

    PreviousLayer previous_layer_;

    alignas(kCacheLineSize)
        BiasType hidden_biases_[kBucketCount * kHiddenDimensions];

    alignas(kCacheLineSize)
        WeightType hidden_weights_[
            kBucketCount * kHiddenDimensions * kPaddedInputDimensions];

    alignas(kCacheLineSize)
        BiasType output_biases_[kBucketCount];

    alignas(kCacheLineSize)
        WeightType output_weights_[kBucketCount * kPaddedHiddenDimensions];
  };

}  // namespace Eval::NNUE::Layers

#endif // NNUE_LAYERS_BUCKETED_TAIL_H_INCLUDED
