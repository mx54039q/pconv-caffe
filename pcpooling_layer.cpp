#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pcpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PCPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(pool_param.has_kernel_size() &&
      !(pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "Filter size is kernel_size, NOT kernel_h and kernel_w";
  }
  CHECK(!(pool_param.has_pad() || 
      pool_param.has_pad_h() || pool_param.has_pad_w()))
      << "No padding!(default set to 0)";
  CHECK(pool_param.has_stride() && 
      !(pool_param.has_stride_h() || pool_param.has_stride_w()))
      << "Stride is stride, NOT stride_h and stride_w.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_ = bottom[0]->channels();
  } else {
    kernel_ = pool_param.kernel_size();
  }
  CHECK_GT(kernel_, 0) << "Filter dimensions cannot be zero.";
  stride_ = pool_param.stride();
  if (global_pooling_) {
    CHECK(pad_ == 0 && stride_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  pad_ = 0;
  if (pad_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_, kernel_);
  }
}

template <typename Dtype>
void PCPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_ = bottom[0]->channels();
  }
  pooled_channels_ = static_cast<int>(ceil(static_cast<float>(
      channels_ + 2 * pad_ - kernel_) / stride_)) + 1;
  if (pad_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_channels_ - 1) * stride_ >= channels_ + pad_) {
      --pooled_channels_;
    }
    CHECK_LT((pooled_channels_ - 1) * stride_, channels_ + pad_);
  }
  top[0]->Reshape(bottom[0]->num(), pooled_channels_, height_,
      width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), pooled_channels_, height_,
        width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), pooled_channels_, height_,
        width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PCPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  const int channel_offset = height_ * width_;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int cstart = pc * stride_;
            int cend = min(cstart + kernel_, channels_);
            const int pool_index = pc * channel_offset + h * width_ + w;
            for (int c = cstart; c < cend; ++c) {
              const int index = c * channel_offset + h * width_ + w;
              if (bottom_data[index] > top_data[pool_index]) {
                top_data[pool_index] = bottom_data[index];
                if (use_top_mask) {
                  top_mask[pool_index] = static_cast<Dtype>(index);
                } else {
                  mask[pool_index] = index;
                }
              }
            }
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
      if (use_top_mask) {
        top_mask += top[0]->offset(1);
      } else {
        mask += top[0]->offset(1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int cstart = pc * stride_;
            int cend = min(cstart + kernel_, channels_);
            int pool_size = cend - cstart;
            const int pool_index = pc * channel_offset + h * width_ + w;
            for (int c = cstart; c < cend; ++c) {
              const int index = c * channel_offset + h * width_ + w;
              top_data[pool_index] += bottom_data[index];
            }
            top_data[pool_index] /= pool_size;
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PCPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  const int channel_offset = height_ * width_;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            const int index = pc * channel_offset + h * width_ + w;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
      }
      // compute offset
      bottom_diff += bottom[0]->offset(1);
      top_diff += top[0]->offset(1);
      if (use_top_mask) {
        top_mask += top[0]->offset(1);
      } else {
        mask += top[0]->offset(1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int cstart = pc * stride_;
            int cend = min(cstart + kernel_, channels_);
            int pool_size = cend - cstart;
            const int top_index = pc * channel_offset + h * width_ + w;
            for (int c = cstart; c < cend; ++c) {
              const int bottom_index = c * channel_offset + h * width_ + w;
              bottom_diff[bottom_index] += top_diff[top_index] / pool_size;
            }
          }
        }
      }
      // compute offset
      bottom_diff += bottom[0]->offset(1);
      top_diff += top[0]->offset(1);
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PCPoolingLayer);
#endif

INSTANTIATE_CLASS(PCPoolingLayer);
REGISTER_LAYER_CLASS(PCPooling);

}  // namespace caffe
