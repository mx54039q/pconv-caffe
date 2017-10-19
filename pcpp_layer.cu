#include <vector>
#include <cfloat>
#include <algorithm>

#include "caffe/layers/pcpp_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2col_gpu_pconv_kernel(const int n, const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;//输出高下标
    const int w_col = index % width_col;//输出宽下标
    const int c_im = h_index / height_col;//输入通道下标
    //const int c_col = c_im * kernel_h * kernel_w;//输出矩阵行下表
    const int h_offset = h_col * stride_h - pad_h;//输入高下标
    const int w_offset = w_col * stride_w - pad_w;//输入宽下标
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_im * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += channels * height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu_pconv(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_pconv_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, channels, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

template void im2col_gpu_pconv<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_col);
template void im2col_gpu_pconv<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, double* data_col);
    
template <typename Dtype>
__global__ void col2im_gpu_pconv_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((h_k * kernel_w + w_k) * channels + c_im) * 
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu_pconv(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_pconv_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu_pconv<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_gpu_pconv<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

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
void PCPPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //Blob 'intermediate' store the data between pconv and pcpooling
  vector<int> inter_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  inter_shape.push_back(conved_channels_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    inter_shape.push_back(this->output_shape_[i]);
  }
  Blob<Dtype>* intermediate = new Blob<Dtype>;
  intermediate->Reshape(inter_shape);
  Dtype* inter_data = intermediate->mutable_gpu_data();
  const int inter_count = intermediate->count();
  caffe_gpu_set(inter_count, Dtype(0.), inter_data);
  
  //Pconv
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();    
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = intermediate->mutable_gpu_data();  
    for (int n = 0; n < (this->num_); ++n) {
      im2col_gpu_pconv(bottom_data + n*this->bottom_dim_, this->channels_,
          height_,width_,
          kernel_shape_data[0], kernel_shape_data[1],
          pad_data[0], pad_data[1],
          stride_data[0], stride_data[1],
          dilation_data[0], dilation_data[1], this->get_mutable_col_buffer_gpu());
      const Dtype* col_buff = this->get_col_buffer_gpu();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_,
        N_, K_,
        (Dtype)1., weight, col_buff,
        (Dtype)0., top_data + n*this->top_dim_);
      if (this->bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(), 
          bias_multiplier_pconv_.gpu_data(), (Dtype)1., top_data + n*this->top_dim_);
      }
    }
  }
  
  //PcPooling
  const Dtype* bottom_data = intermediate->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pcpp_param().pool()) {
  case PcppParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), conved_channels_,
        height_out_, width_out_, pooled_channels_, pool_kernel_,
        pool_stride_, pool_pad_, top_data, mask, top_mask);
    break;
  case PcppParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), conved_channels_,
        height_out_, width_out_, pooled_channels_, pool_kernel_,
        pool_stride_, pool_pad_,  top_data);
    break;
  case PcppParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), conved_channels_,
        height_out_, width_out_, pooled_channels_, pool_kernel_,
        pool_stride_, rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), conved_channels_,
        height_out_, width_out_, pooled_channels_, pool_kernel_,
        pool_stride_, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  delete intermediate;
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
void PCPPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //Blob 'intermediate' store the data between pconv and pcpooling
  vector<int> inter_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  inter_shape.push_back(conved_channels_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    inter_shape.push_back(this->output_shape_[i]);
  }
  Blob<Dtype>* intermediate = new Blob<Dtype>;
  intermediate->Reshape(inter_shape);
  
  //PcPooling
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = intermediate->mutable_gpu_diff();
  const int count = intermediate->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pcpp_param().pool()) {
  case PcppParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), conved_channels_,
        height_out_, width_out_, pooled_channels_, pool_kernel_,
        pool_stride_, pool_pad_, bottom_diff);
    break;
  case PcppParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), conved_channels_,
        height_out_, width_out_, pooled_channels_, pool_kernel_,
        pool_stride_, pool_pad_, bottom_diff);
    break;
  case PcppParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff, top[0]->num(), 
        conved_channels_, height_out_, width_out_, pooled_channels_, pool_kernel_,
        pool_stride_, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  
  //Pconv
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    top_diff = intermediate->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, M_, N_,
          1., top_diff + n*this->top_dim_, bias_multiplier_pconv_.gpu_data(), 1., bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        im2col_gpu_pconv(bottom_data + n*this->bottom_dim_, this->channels_,
          height_,width_,
          kernel_shape_data[0], kernel_shape_data[1],
          pad_data[0], pad_data[1],
          stride_data[0], stride_data[1],
          dilation_data[0], dilation_data[1], this->get_mutable_col_buffer_gpu());
        const Dtype* col_buff = this->get_col_buffer_gpu();
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_,
            K_, N_,
            (Dtype)1., top_diff + n * this->top_dim_, col_buff,
            (Dtype)1., weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_,
            N_, M_,
            (Dtype)1., weight, top_diff + n * this->top_dim_,
            (Dtype)0., this->get_mutable_col_buffer_gpu());
        }
        col2im_gpu_pconv(col_buff, this->channels_,
          height_,width_,
          kernel_shape_data[0], kernel_shape_data[1],
          pad_data[0], pad_data[1],
          stride_data[0], stride_data[1],
          dilation_data[0], dilation_data[1], bottom_diff + n*this->bottom_dim_);
      }
    }
  }
  delete intermediate;
  //this->num_output_ = temp_num_output;
}



INSTANTIATE_LAYER_GPU_FUNCS(PCPPLayer);

}  // namespace caffe
