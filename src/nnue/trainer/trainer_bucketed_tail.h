#ifndef NNUE_TRAINER_BUCKETED_TAIL_H_INCLUDED
#define NNUE_TRAINER_BUCKETED_TAIL_H_INCLUDED

#include "trainer.h"

#include "learn/learn.h"

#include "nnue/layers/bucketed_tail.h"

#include "thread.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Specialization of NNUE evaluation function learning class template for BucketedTail
namespace Eval::NNUE {

    // Learning: experimental bucketed 32 -> 32 -> 1 tail
    template <typename PreviousLayer, IndexType BucketCount, IndexType HiddenDimensions>
    class Trainer<Layers::BucketedTail<PreviousLayer, BucketCount, HiddenDimensions>> {
    private:
        // Type of layer to learn
        using LayerType = Layers::BucketedTail<PreviousLayer, BucketCount, HiddenDimensions>;

    public:
        // factory function
        static std::shared_ptr<Trainer> create(
            LayerType* target_layer, FeatureTransformer* ft) {

            return std::shared_ptr<Trainer>(
                new Trainer(target_layer, ft));
        }

        // Set options such as hyperparameters
        void send_message(Message* message) {
            previous_layer_trainer_->send_message(message);

            if (receive_message("momentum", message)) {
                momentum_ = static_cast<LearnFloatType>(std::stod(message->value));
            }

            if (receive_message("learning_rate_scale", message)) {
                learning_rate_scale_ =
                    static_cast<LearnFloatType>(std::stod(message->value));
            }

            if (receive_message("reset", message)) {
                dequantize_parameters();
            }

            if (receive_message("quantize_parameters", message)) {
                quantize_parameters();
            }

            if (receive_message("check_health", message)) {
                check_health();
            }
        }

        // Initialize the parameters with random numbers
        template <typename RNG>
        void initialize(RNG& rng) {
            previous_layer_trainer_->initialize(rng);

            // Hidden tail layer:
            // Initialize bucket 0 like a normal non-output affine layer, then copy
            // the same initial tail into the other buckets. This gives all buckets
            // the same trained starting point before real bucket specialization.
            const double hidden_sigma = 1.0 / std::sqrt(kInputDimensions);
            auto hidden_distribution =
                std::normal_distribution<double>(0.0, hidden_sigma);

            for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                double sum = 0.0;

                const IndexType weight_offset = h * kInputDimensions;

                for (IndexType i = 0; i < kInputDimensions; ++i) {
                    const auto weight =
                        static_cast<LearnFloatType>(hidden_distribution(rng));
                    hidden_weights_[weight_offset + i] = weight;
                    sum += weight;
                }

                hidden_biases_[h] =
                    static_cast<LearnFloatType>(0.5 - 0.5 * sum);
            }

            // Copy bucket 0 hidden parameters to buckets 1..N.
            for (IndexType bucket = 1; bucket < kBucketCount; ++bucket) {
                const IndexType target_bias_offset =
                    bucket * kHiddenDimensions;

                const IndexType target_weight_offset =
                    bucket * kHiddenDimensions * kInputDimensions;

                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    hidden_biases_[target_bias_offset + h] =
                        hidden_biases_[h];

                    const IndexType source_weight_offset =
                        h * kInputDimensions;

                    const IndexType target_row_offset =
                        target_weight_offset + h * kInputDimensions;

                    for (IndexType i = 0; i < kInputDimensions; ++i) {
                        hidden_weights_[target_row_offset + i] =
                            hidden_weights_[source_weight_offset + i];
                    }
                }
            }

            // Output tail layer:
            // Initialize all buckets identically to zero, matching the normal output layer.
            std::fill(std::begin(output_biases_), std::end(output_biases_),
                      static_cast<LearnFloatType>(0.0));

            std::fill(std::begin(output_weights_), std::end(output_weights_),
                      static_cast<LearnFloatType>(0.0));

            quantize_parameters();
        }

        const LearnFloatType* step_start(
            ThreadPool& thread_pool,
            std::vector<Example>::const_iterator batch_begin,
            std::vector<Example>::const_iterator batch_end) {

            const auto size = batch_end - batch_begin;

            if ((long)output_.size() < (long)kOutputDimensions * size) {
                output_.resize(kOutputDimensions * size);
                hidden_pre_activations_.resize(kHiddenDimensions * size);
                hidden_outputs_.resize(kHiddenDimensions * size);
                gradients_.resize(kInputDimensions * size);
                bucket_indices_.resize(size);
            }

            if (thread_states_.size() < thread_pool.size()) {
                thread_states_.resize(thread_pool.size());
            }

            batch_size_ = size;
            combined_batch_input_ =
                previous_layer_trainer_->step_start(thread_pool, batch_begin, batch_end);

            // Selected-bucket training:
            // forward output and backpropagation use the bucket stored on each example.
            for (IndexType b = 0; b < size; ++b) {
                const IndexType bucket = (batch_begin + b)->bucket_index;
                assert(bucket < kBucketCount);
                bucket_indices_[b] = bucket;
            }

            scale_thread_state(thread_states_[0], momentum_);

            for (IndexType i = 1; i < thread_states_.size(); ++i) {
                thread_states_[i].reset();
            }

            return output_.data();
        }

        // forward propagation
        void propagate(Thread& th, const uint64_t offset, const uint64_t count) {

            previous_layer_trainer_->propagate(th, offset, count);

            for (IndexType b = offset; b < offset + count; ++b) {
                const IndexType bucket = bucket_indices_[b];

                const IndexType input_offset = b * kInputDimensions;
                const IndexType hidden_offset = b * kHiddenDimensions;

                const IndexType hidden_bias_offset =
                    bucket * kHiddenDimensions;

                const IndexType hidden_weight_bucket_offset =
                    bucket * kHiddenDimensions * kInputDimensions;

                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    LearnFloatType sum =
                        hidden_biases_[hidden_bias_offset + h];

                    const IndexType weight_offset =
                        hidden_weight_bucket_offset + h * kInputDimensions;

                    for (IndexType i = 0; i < kInputDimensions; ++i) {
                        sum += hidden_weights_[weight_offset + i]
                            * combined_batch_input_[input_offset + i];
                    }

                    hidden_pre_activations_[hidden_offset + h] = sum;
                    hidden_outputs_[hidden_offset + h] =
                        std::max(kZero, std::min(kOne, sum));
                }

                LearnFloatType sum = output_biases_[bucket];

                const IndexType output_weight_offset =
                    bucket * kHiddenDimensions;

                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    sum += output_weights_[output_weight_offset + h]
                        * hidden_outputs_[hidden_offset + h];
                }

                output_[b * kOutputDimensions] = sum;
            }
        }

        // backpropagation
        void backpropagate(
            Thread& th,
            const LearnFloatType* gradients,
            uint64_t offset,
            uint64_t count) {

            auto& thread_state = thread_states_[th.thread_idx()];

            // Selected-bucket mode:
            // Use the bucket selected for each training example.
            // Forward propagation used the same bucket index in step_start(),
            // and backpropagation updates only that bucket tail.
            // The shared previous layer receives the gradient from the selected tail only.
            for (IndexType b = offset; b < offset + count; ++b) {
                const IndexType bucket = bucket_indices_[b];

                const IndexType input_offset = b * kInputDimensions;
                const IndexType hidden_offset = b * kHiddenDimensions;

                // Clear gradient to previous shared layer for this sample.
                for (IndexType i = 0; i < kInputDimensions; ++i) {
                    gradients_[input_offset + i] = static_cast<LearnFloatType>(0.0);
                }

                const LearnFloatType output_gradient =
                    gradients[b * kOutputDimensions];

                const IndexType hidden_bias_offset =
                    bucket * kHiddenDimensions;

                const IndexType hidden_weight_bucket_offset =
                    bucket * kHiddenDimensions * kInputDimensions;

                const IndexType output_weight_offset =
                    bucket * kHiddenDimensions;

                thread_state.output_biases_diff_[bucket] += output_gradient;

                // Gradient through selected bucket's output affine.
                LearnFloatType hidden_gradients[kHiddenDimensions];

                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    const LearnFloatType hidden_output =
                        hidden_outputs_[hidden_offset + h];

                    thread_state.output_weights_diff_[output_weight_offset + h] +=
                        output_gradient * hidden_output;

                    hidden_gradients[h] =
                        output_gradient * output_weights_[output_weight_offset + h];
                }

                // Gradient through selected bucket's clipped ReLU and hidden affine.
                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    const LearnFloatType hidden_output =
                        hidden_outputs_[hidden_offset + h];

                    const bool clipped =
                        (hidden_output <= kZero) || (hidden_output >= kOne);

                    const LearnFloatType hidden_gradient =
                        clipped ? static_cast<LearnFloatType>(0.0)
                                : hidden_gradients[h];

                    thread_state.hidden_biases_diff_[hidden_bias_offset + h] +=
                        hidden_gradient;

                    const IndexType weight_offset =
                        hidden_weight_bucket_offset + h * kInputDimensions;

                    for (IndexType i = 0; i < kInputDimensions; ++i) {
                        thread_state.hidden_weights_diff_[weight_offset + i] +=
                            hidden_gradient * combined_batch_input_[input_offset + i];

                        gradients_[input_offset + i] +=
                            hidden_gradient * hidden_weights_[weight_offset + i];
                    }
                }
            }

            previous_layer_trainer_->backpropagate(
                th, gradients_.data(), offset, count);
        }

        void reduce_thread_state() {
            for (IndexType i = 1; i < thread_states_.size(); ++i) {
                thread_states_[0] += thread_states_[i];
            }
        }

        void step_end(ThreadPool& thread_pool, LearnFloatType learning_rate) {
            const LearnFloatType local_learning_rate =
                learning_rate * learning_rate_scale_;

            reduce_thread_state();

            auto& main_thread_state = thread_states_[0];

            for (IndexType i = 0; i < kHiddenBiasCount; ++i) {
                const double d =
                    local_learning_rate * main_thread_state.hidden_biases_diff_[i];
                hidden_biases_[i] -= d;
                abs_biases_diff_sum_ += std::abs(d);
            }

            for (IndexType i = 0; i < kHiddenWeightCount; ++i) {
                const double d =
                    local_learning_rate * main_thread_state.hidden_weights_diff_[i];
                hidden_weights_[i] -= d;
                abs_weights_diff_sum_ += std::abs(d);
            }

            for (IndexType i = 0; i < kOutputBiasCount; ++i) {
                const double d =
                    local_learning_rate * main_thread_state.output_biases_diff_[i];
                output_biases_[i] -= d;
                abs_biases_diff_sum_ += std::abs(d);
            }

            for (IndexType i = 0; i < kOutputWeightCount; ++i) {
                const double d =
                    local_learning_rate * main_thread_state.output_weights_diff_[i];
                output_weights_[i] -= d;
                abs_weights_diff_sum_ += std::abs(d);
            }

            num_biases_diffs_ += kHiddenBiasCount + kOutputBiasCount;
            num_weights_diffs_ += kHiddenWeightCount + kOutputWeightCount;

            previous_layer_trainer_->step_end(thread_pool, learning_rate);
        }

    private:
        // constructor
        Trainer(LayerType* target_layer, FeatureTransformer* ft) :
            batch_size_(0),
            combined_batch_input_(nullptr),
            previous_layer_trainer_(Trainer<PreviousLayer>::create(
                &target_layer->previous_layer_, ft)),
            target_layer_(target_layer),
            momentum_(0.9),
            learning_rate_scale_(1.0) {

            dequantize_parameters();
        }

        void reset_stats() {
            abs_biases_diff_sum_ = 0.0;
            abs_weights_diff_sum_ = 0.0;
            num_biases_diffs_ = 0;
            num_weights_diffs_ = 0;
        }

        void check_health() {
            double abs_bias_sum = 0.0;
            double abs_weight_sum = 0.0;

            for (auto b : hidden_biases_)
                abs_bias_sum += std::abs(b);

            for (auto b : output_biases_)
                abs_bias_sum += std::abs(b);

            for (auto w : hidden_weights_)
                abs_weight_sum += std::abs(w);

            for (auto w : output_weights_)
                abs_weight_sum += std::abs(w);

            const double bias_count =
                static_cast<double>(kHiddenBiasCount + kOutputBiasCount);

            const double weight_count =
                static_cast<double>(kHiddenWeightCount + kOutputWeightCount);

            const double avg_bias_diff =
                num_biases_diffs_ == 0
                    ? 0.0
                    : abs_biases_diff_sum_ / num_biases_diffs_;

            const double avg_weight_diff =
                num_weights_diffs_ == 0
                    ? 0.0
                    : abs_weights_diff_sum_ / num_weights_diffs_;

            double hidden_bias_bucket_delta_sum = 0.0;
            double hidden_weight_bucket_delta_sum = 0.0;
            double output_bias_bucket_delta_sum = 0.0;
            double output_weight_bucket_delta_sum = 0.0;

            double hidden_bias_bucket_delta_max = 0.0;
            double hidden_weight_bucket_delta_max = 0.0;
            double output_bias_bucket_delta_max = 0.0;
            double output_weight_bucket_delta_max = 0.0;

            for (IndexType bucket = 1; bucket < kBucketCount; ++bucket) {
                const IndexType hidden_bias_offset =
                    bucket * kHiddenDimensions;

                const IndexType hidden_weight_offset =
                    bucket * kHiddenDimensions * kInputDimensions;

                const IndexType output_weight_offset =
                    bucket * kHiddenDimensions;

                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    const double hidden_bias_delta = std::abs(
                        hidden_biases_[hidden_bias_offset + h]
                        - hidden_biases_[h]);

                    hidden_bias_bucket_delta_sum += hidden_bias_delta;
                    hidden_bias_bucket_delta_max =
                        std::max(hidden_bias_bucket_delta_max, hidden_bias_delta);

                    const double output_weight_delta = std::abs(
                        output_weights_[output_weight_offset + h]
                        - output_weights_[h]);

                    output_weight_bucket_delta_sum += output_weight_delta;
                    output_weight_bucket_delta_max =
                        std::max(output_weight_bucket_delta_max, output_weight_delta);

                    const IndexType bucket_row_offset =
                        hidden_weight_offset + h * kInputDimensions;

                    const IndexType bucket0_row_offset =
                        h * kInputDimensions;

                    for (IndexType i = 0; i < kInputDimensions; ++i) {
                        const double hidden_weight_delta = std::abs(
                            hidden_weights_[bucket_row_offset + i]
                            - hidden_weights_[bucket0_row_offset + i]);

                        hidden_weight_bucket_delta_sum += hidden_weight_delta;
                        hidden_weight_bucket_delta_max =
                            std::max(hidden_weight_bucket_delta_max, hidden_weight_delta);
                    }
                }

                const double output_bias_delta = std::abs(
                    output_biases_[bucket] - output_biases_[0]);

                output_bias_bucket_delta_sum += output_bias_delta;
                output_bias_bucket_delta_max =
                    std::max(output_bias_bucket_delta_max, output_bias_delta);
            }

            const double compared_bucket_count =
                static_cast<double>(kBucketCount > 1 ? kBucketCount - 1 : 0);

            const double hidden_bias_bucket_delta_count =
                compared_bucket_count * kHiddenDimensions;

            const double hidden_weight_bucket_delta_count =
                compared_bucket_count * kHiddenDimensions * kInputDimensions;

            const double output_bias_bucket_delta_count =
                compared_bucket_count;

            const double output_weight_bucket_delta_count =
                compared_bucket_count * kHiddenDimensions;

            const double hidden_bias_bucket_delta_avg =
                hidden_bias_bucket_delta_count == 0.0
                    ? 0.0
                    : hidden_bias_bucket_delta_sum / hidden_bias_bucket_delta_count;

            const double hidden_weight_bucket_delta_avg =
                hidden_weight_bucket_delta_count == 0.0
                    ? 0.0
                    : hidden_weight_bucket_delta_sum / hidden_weight_bucket_delta_count;

            const double output_bias_bucket_delta_avg =
                output_bias_bucket_delta_count == 0.0
                    ? 0.0
                    : output_bias_bucket_delta_sum / output_bias_bucket_delta_count;

            const double output_weight_bucket_delta_avg =
                output_weight_bucket_delta_count == 0.0
                    ? 0.0
                    : output_weight_bucket_delta_sum / output_weight_bucket_delta_count;

            auto out = sync_region_cout.new_region();

            out << "INFO (check_health):"
                << " layer " << LayerType::kLayerIndex
                << " - " << LayerType::get_name()
                << std::endl;

            out << "  - avg_abs_bias        = " << abs_bias_sum / bias_count << std::endl;
            out << "  - avg_abs_bias_diff   = " << avg_bias_diff << std::endl;
            out << "  - avg_abs_weight      = " << abs_weight_sum / weight_count << std::endl;
            out << "  - avg_abs_weight_diff = " << avg_weight_diff << std::endl;

            out << "  - bucket mode         = selected bucket used for forward output and backprop" << std::endl;
            out << "  - bucket delta vs 0   = hidden_bias max " << hidden_bias_bucket_delta_max
                << " avg " << hidden_bias_bucket_delta_avg << std::endl;
            out << "  - bucket delta vs 0   = hidden_weight max " << hidden_weight_bucket_delta_max
                << " avg " << hidden_weight_bucket_delta_avg << std::endl;
            out << "  - bucket delta vs 0   = output_bias max " << output_bias_bucket_delta_max
                << " avg " << output_bias_bucket_delta_avg << std::endl;
            out << "  - bucket delta vs 0   = output_weight max " << output_weight_bucket_delta_max
                << " avg " << output_weight_bucket_delta_avg << std::endl;

            out.unlock();

            reset_stats();
        }

        void quantize_parameters() {
            for (IndexType i = 0; i < kHiddenWeightCount; ++i) {
                hidden_weights_[i] =
                    std::max(-kMaxHiddenWeightMagnitude,
                             std::min(+kMaxHiddenWeightMagnitude, hidden_weights_[i]));
            }

            for (IndexType i = 0; i < kOutputWeightCount; ++i) {
                output_weights_[i] =
                    std::max(-kMaxOutputWeightMagnitude,
                             std::min(+kMaxOutputWeightMagnitude, output_weights_[i]));
            }

            for (IndexType bucket = 0; bucket < kBucketCount; ++bucket) {
                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    const IndexType index =
                        bucket * kHiddenDimensions + h;

                    target_layer_->hidden_biases_[index] =
                        round<std::int32_t>(hidden_biases_[index] * kHiddenBiasScale);
                }
            }

            for (IndexType bucket = 0; bucket < kBucketCount; ++bucket) {
                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    const IndexType source_offset =
                        bucket * kHiddenDimensions * kInputDimensions
                        + h * kInputDimensions;

                    const IndexType target_offset =
                        bucket * kHiddenDimensions * LayerType::kPaddedInputDimensions
                        + h * LayerType::kPaddedInputDimensions;

                    for (IndexType i = 0; i < kInputDimensions; ++i) {
                        target_layer_->hidden_weights_[target_offset + i] =
                            round<std::int8_t>(
                                hidden_weights_[source_offset + i] * kHiddenWeightScale);
                    }
                }
            }

            for (IndexType bucket = 0; bucket < kBucketCount; ++bucket) {
                target_layer_->output_biases_[bucket] =
                    round<std::int32_t>(output_biases_[bucket] * kOutputBiasScale);
            }

            for (IndexType bucket = 0; bucket < kBucketCount; ++bucket) {
                const IndexType source_offset =
                    bucket * kHiddenDimensions;

                const IndexType target_offset =
                    bucket * LayerType::kPaddedHiddenDimensions;

                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    target_layer_->output_weights_[target_offset + h] =
                        round<std::int8_t>(
                            output_weights_[source_offset + h] * kOutputWeightScale);
                }
            }
        }

        void dequantize_parameters() {
            for (IndexType bucket = 0; bucket < kBucketCount; ++bucket) {
                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    const IndexType index =
                        bucket * kHiddenDimensions + h;

                    hidden_biases_[index] =
                        static_cast<LearnFloatType>(
                            target_layer_->hidden_biases_[index] / kHiddenBiasScale);
                }
            }

            for (IndexType bucket = 0; bucket < kBucketCount; ++bucket) {
                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    const IndexType target_offset =
                        bucket * kHiddenDimensions * LayerType::kPaddedInputDimensions
                        + h * LayerType::kPaddedInputDimensions;

                    const IndexType source_offset =
                        bucket * kHiddenDimensions * kInputDimensions
                        + h * kInputDimensions;

                    for (IndexType i = 0; i < kInputDimensions; ++i) {
                        hidden_weights_[source_offset + i] =
                            static_cast<LearnFloatType>(
                                target_layer_->hidden_weights_[target_offset + i]
                                / kHiddenWeightScale);
                    }
                }
            }

            for (IndexType bucket = 0; bucket < kBucketCount; ++bucket) {
                output_biases_[bucket] =
                    static_cast<LearnFloatType>(
                        target_layer_->output_biases_[bucket] / kOutputBiasScale);
            }

            for (IndexType bucket = 0; bucket < kBucketCount; ++bucket) {
                const IndexType target_offset =
                    bucket * LayerType::kPaddedHiddenDimensions;

                const IndexType source_offset =
                    bucket * kHiddenDimensions;

                for (IndexType h = 0; h < kHiddenDimensions; ++h) {
                    output_weights_[source_offset + h] =
                        static_cast<LearnFloatType>(
                            target_layer_->output_weights_[target_offset + h]
                            / kOutputWeightScale);
                }
            }

            for (auto& state : thread_states_) {
                state.reset();
            }

            reset_stats();
        }

        // number of input/output dimensions
        static constexpr IndexType kInputDimensions = LayerType::kInputDimensions;
        static constexpr IndexType kHiddenDimensions = LayerType::kHiddenDimensions;
        static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;
        static constexpr IndexType kBucketCount = LayerType::kBucketCount;

        static_assert(kOutputDimensions == 1, "");

        static constexpr IndexType kHiddenBiasCount =
            kBucketCount * kHiddenDimensions;

        static constexpr IndexType kHiddenWeightCount =
            kBucketCount * kHiddenDimensions * kInputDimensions;

        static constexpr IndexType kOutputBiasCount =
            kBucketCount;

        static constexpr IndexType kOutputWeightCount =
            kBucketCount * kHiddenDimensions;

        // LearnFloatType constant
        static constexpr LearnFloatType kZero = static_cast<LearnFloatType>(0.0);
        static constexpr LearnFloatType kOne = static_cast<LearnFloatType>(1.0);

        // Coefficients used for parameterization
        static constexpr LearnFloatType kActivationScale =
            std::numeric_limits<std::int8_t>::max();

        // Hidden 32 -> 32 layer uses normal hidden-layer scaling.
        static constexpr LearnFloatType kHiddenBiasScale =
            (1 << kWeightScaleBits) * kActivationScale;

        static constexpr LearnFloatType kHiddenWeightScale =
            kHiddenBiasScale / kActivationScale;

        // Output 32 -> 1 layer uses normal output-layer scaling.
        static constexpr LearnFloatType kOutputBiasScale =
            kPonanzaConstant * FV_SCALE;

        static constexpr LearnFloatType kOutputWeightScale =
            kOutputBiasScale / kActivationScale;

        static constexpr LearnFloatType kMaxHiddenWeightMagnitude =
            std::numeric_limits<std::int8_t>::max() / kHiddenWeightScale;

        static constexpr LearnFloatType kMaxOutputWeightMagnitude =
            std::numeric_limits<std::int8_t>::max() / kOutputWeightScale;

        struct alignas(kCacheLineSize) ThreadState {
            alignas(kCacheLineSize)
                LearnFloatType hidden_biases_diff_[kHiddenBiasCount];

            alignas(kCacheLineSize)
                LearnFloatType hidden_weights_diff_[kHiddenWeightCount];

            alignas(kCacheLineSize)
                LearnFloatType output_biases_diff_[kOutputBiasCount];

            alignas(kCacheLineSize)
                LearnFloatType output_weights_diff_[kOutputWeightCount];

            ThreadState() {
                reset();
            }

            ThreadState& operator+=(const ThreadState& other) {
                for (IndexType i = 0; i < kHiddenBiasCount; ++i)
                    hidden_biases_diff_[i] += other.hidden_biases_diff_[i];

                for (IndexType i = 0; i < kHiddenWeightCount; ++i)
                    hidden_weights_diff_[i] += other.hidden_weights_diff_[i];

                for (IndexType i = 0; i < kOutputBiasCount; ++i)
                    output_biases_diff_[i] += other.output_biases_diff_[i];

                for (IndexType i = 0; i < kOutputWeightCount; ++i)
                    output_weights_diff_[i] += other.output_weights_diff_[i];

                return *this;
            }

            void reset() {
                std::fill(std::begin(hidden_biases_diff_),
                          std::end(hidden_biases_diff_), 0.0f);

                std::fill(std::begin(hidden_weights_diff_),
                          std::end(hidden_weights_diff_), 0.0f);

                std::fill(std::begin(output_biases_diff_),
                          std::end(output_biases_diff_), 0.0f);

                std::fill(std::begin(output_weights_diff_),
                          std::end(output_weights_diff_), 0.0f);
            }
        };

        static void scale_thread_state(ThreadState& state, LearnFloatType scale) {
            for (auto& v : state.hidden_biases_diff_)
                v *= scale;

            for (auto& v : state.hidden_weights_diff_)
                v *= scale;

            for (auto& v : state.output_biases_diff_)
                v *= scale;

            for (auto& v : state.output_weights_diff_)
                v *= scale;
        }

        // number of samples in mini-batch
        IndexType batch_size_;

        double abs_biases_diff_sum_;
        double abs_weights_diff_sum_;
        uint64_t num_biases_diffs_;
        uint64_t num_weights_diffs_;

        // Input mini batch
        const LearnFloatType* combined_batch_input_;

        // Trainer of the previous layer
        const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

        // layer to learn
        LayerType* const target_layer_;

        // Float-domain parameters
        alignas(kCacheLineSize)
            LearnFloatType hidden_biases_[kHiddenBiasCount];

        alignas(kCacheLineSize)
            LearnFloatType hidden_weights_[kHiddenWeightCount];

        alignas(kCacheLineSize)
            LearnFloatType output_biases_[kOutputBiasCount];

        alignas(kCacheLineSize)
            LearnFloatType output_weights_[kOutputWeightCount];

        std::vector<ThreadState, CacheLineAlignedAllocator<ThreadState>> thread_states_;

        // Forward propagation buffers
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> output_;
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> hidden_pre_activations_;
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> hidden_outputs_;

        // Buffer for back propagation to shared previous layer
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> gradients_;

        // Selected bucket indices for the current mini-batch.
        std::vector<IndexType> bucket_indices_;

        // hyper parameter
        LearnFloatType momentum_;
        LearnFloatType learning_rate_scale_;
    };

}  // namespace Eval::NNUE

#endif
