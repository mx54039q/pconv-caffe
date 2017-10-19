#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pcpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_channels,
    const int kernel, const int stride, const int pad,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int pc = (index / width / height) % pooled_channels;
    const int n = index / width / height / pooled_channels;
    int cstart = pc * stride - pad;
    int cend = min(cstart + kernel, channels);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels * height + h) * width + w;
    for (int c = cstart; c < cend; ++c) {
      if (bottom_slice[c * height * width] > maxval) {
        maxidx = c * height * width;
        maxval = bottom_slice[maxidx];
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_channels,
    const int kernel, const int stride, const int pad,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int pc = (index / width / height) % pooled_channels;
    const int n = index / width / height / pooled_channels;
    int cstart = pc * stride - pad;
    int cend = min(cstart + kernel, channels);
    const int pool_size = cend - cstart;
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels * height + h) * width + w;
    for (int c = cstart; c < cend; ++c) {
      aveval += bottom_slice[c * height * width];
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_channels,
    const int kernel, const int stride, Dtype* const rand_idx, 
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int pc = (index / width / height) % pooled_channels;
    const int n = index / width / height / pooled_channels;
    int cstart = pc * stride;
    int cend = min(cstart + kernel, channels);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels * height + h) * width + w;
    // First pass: get sum
    for (int c = cstart; c < cend; ++c) {
      cumsum += bottom_slice[c * height * width];
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int c = cstart; c < cend; ++c) {
      cumsum += bottom_slice[c * height * width];
      if (cumsum >= thres) {
        rand_idx[index] = ((n * channels + c) * height + h) * width + w;
        top_data[index] = bottom_slice[c * height * width];
        return;
      }
    }
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_channels,
    const int kernel, const int stride, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int pc = (index / width / height) % pooled_channels;
    const int n = index / width / height / pooled_channels;
    int cstart = pc * stride;
    int cend = min(cstart + kernel, channels);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = 0.;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels * height + h) * width + w;
    // First pass: get sum
    for (int c = cstart; c < cend; ++c) {
      cumsum += bottom_slice[c * height * width];
      cumvalues += bottom_slice[c * height * width] * bottom_slice[c * height * width];
    }
    top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.;
  }
}

template <typename Dtype>
void PCPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_channels_, kernel_,
        stride_, pad_, top_data, mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_channels_, kernel_,
        stride_, pad_,  top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_channels_, kernel_,
          stride_, rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_channels_, kernel_,
          stride_, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_channels, const int kernel,
    const int stride,const int pad, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int pcstart = (c + pad < kernel) ? 0 : (c + pad - kernel) / stride + 1;
    const int pcend = min((c + pad) / stride + 1, pooled_channels);
    Dtype gradient = 0;
    const int top_offset = (n * pooled_channels * height + h) * width + w;
    const Dtype* const top_diff_slice = top_diff + top_offset;
    if (mask) {
      const int* const mask_slice = mask + top_offset;
      for (int pc = pcstart; pc < pcend; ++pc) {
        if (mask_slice[pc * width * height] == c * width * height) {
          gradient += top_diff_slice[pc * width * height];
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + top_offset;
      for (int pc = pcstart; pc < pcend; ++pc) {
        if (top_mask_slice[pc * width * height] == c * height * width) {
          gradient += top_diff_slice[pc * width * height];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int pooled_channels, const int kernel,
    const int stride,const int pad, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int pcstart = (c + pad < kernel) ? 0 : (c + pad - kernel) / stride + 1;
    const int pcend = min((c + pad) / stride + 1, pooled_channels);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * pooled_channels * height + h) * width + w;;
    for (int pc = pcstart; pc < pcend; ++pc) {
        // figure out the pooling size
        int cstart = pc * stride - pad;
        int cend = min(cstart + kernel, channels + pad);
        int pool_size = cend - cstart;
        gradient += top_diff_slice[pc * width * height] / pool_size;
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads, 
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int pooled_channels, const int kernel,
    const int stride, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int pcstart = (c < kernel) ? 0 : (c - kernel) / stride + 1;
    const int pcend = min(c / stride + 1, pooled_channels);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * pooled_channels * height + h) * width + w;;
    const Dtype* const top_diff_slice =
        top_diff + (n * pooled_channels * height + h) * width + w;;
    for (int pc = pcstart; pc < pcend; ++pc) {
      gradient += top_diff_slice[pc * width * height] *
          (index == static_cast<int>(rand_idx_slice[pc * width * height]));
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void PCPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_channels_,
        kernel_, stride_, pad_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_channels_,
        kernel_, stride_, pad_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff, top[0]->num(), 
        channels_, height_, width_, pooled_channels_,
        kernel_, stride_, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PCPoolingLayer);


}  // namespace caffe
