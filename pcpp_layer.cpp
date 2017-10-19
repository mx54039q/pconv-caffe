#include <algorithm>
#include <cfloat>
#include <vector>


#include "caffe/layers/pcpp_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

//
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void PCPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PcppParameter pcpp_param = this->layer_param_.pcpp_param();
  CHECK_EQ(bottom.size(), 1) << "PCPP Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "PCPPLayer Layer takes a single blob as output.";
    
  //********Set Weight & Bias Parameters of Pconv********
  // Setup filter kernel dimensions (kernel_shape_).
  this->channel_axis_ = bottom[0]->CanonicalAxisIndex(pcpp_param.axis());
  this->num_spatial_axes_ = 2;
  vector<int> spatial_dim_blob_shape(1, 2);
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  if (pcpp_param.has_kernel_h() || pcpp_param.has_kernel_w()) {
    CHECK_EQ(0, pcpp_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = pcpp_param.kernel_h();
    kernel_shape_data[1] = pcpp_param.kernel_w();
  } else {
    const int num_kernel_dims = pcpp_param.kernel_size_size();
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      kernel_shape_data[i] =
          pcpp_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  for (int i = 0; i < 2; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);
  int* conv_stride_data = this->stride_.mutable_cpu_data();
  if (pcpp_param.has_stride_h() || pcpp_param.has_stride_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, pcpp_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    conv_stride_data[0] = pcpp_param.stride_h();
    conv_stride_data[1] = pcpp_param.stride_w();
  } else {
    const int num_kernel_dims = pcpp_param.kernel_size_size();
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
        conv_stride_data[i] =
            pcpp_param.stride((num_kernel_dims == 1) ? 0 : i);
    }
  }
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* conv_pad_data = this->pad_.mutable_cpu_data();
  if (pcpp_param.has_pad_h() || pcpp_param.has_pad_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, pcpp_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    conv_pad_data[0] = pcpp_param.pad_h();
    conv_pad_data[1] = pcpp_param.pad_w();
  } else {
    const int num_pad_dims = pcpp_param.pad_size();
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      conv_pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          pcpp_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  this->dilation_.Reshape(spatial_dim_blob_shape);
  int* conv_dilation_data = this->dilation_.mutable_cpu_data();
  const int num_dilation_dims = pcpp_param.dilation_size();
  const int kDefaultDilation = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    conv_dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       pcpp_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    this->is_1x1_ &=
        kernel_shape_data[i] == 1 && conv_stride_data[i] == 1 && conv_pad_data[i] == 0;
    if (!this->is_1x1_) { break; }
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(this->channel_axis_);
  this->group_ = pcpp_param.group();
  CHECK_EQ(this->group_, 1) << "Multi Group NOT Implement.";
  //Set Pconv Parameters
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  const int extent_h = dilation_data[0] * (kernel_shape_data[0] - 1) + 1;
  height_out_ = (height_ + 2 * pad_data[0] - extent_h) /
                stride_data[0] + 1;
  const int extent_w = dilation_data[1] * (kernel_shape_data[1] - 1) + 1;
  width_out_ = (width_ + 2 * pad_data[1] - extent_w) /
                stride_data[1] + 1;             
  M_ = pcpp_param.kernel_num(); // number of kernel;
  N_ = this->channels_ * height_out_ * width_out_;
  K_ = kernel_shape_data[0]*kernel_shape_data[1];
  this->num_output_ = this->channels_ * M_;
  // Intialize the weight&bias and fill the weight&bias
  vector<int> weight_shape(2); 
  weight_shape[0] = M_;
  weight_shape[1] = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {  
    weight_shape.push_back(kernel_shape_data[i]);
  }
  this->bias_term_ = pcpp_param.bias_term();
  vector<int> bias_shape(this->bias_term_, M_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (this->bias_term_) {  
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(  
          this->layer_param_.convolution_param().bias_filler()));  
      bias_filler->Fill(this->blobs_[1].get());//fill the bias
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  this->set_kernel_dim_(this->channels_ * K_);
  
  
  //*******Set PcPooling Parameters*******
  if (pcpp_param.global_pooling()) {
    CHECK(!pcpp_param.has_pool_kernel_size())
      << "With Global_pooling: true Filter size cannot specified";
  }
  global_pooling_ = pcpp_param.global_pooling();
  if (global_pooling_) {
    pool_kernel_ = bottom[0]->channels();
  } else {
    pool_kernel_ = pcpp_param.pool_kernel_size();
  }
  CHECK_GT(pool_kernel_, 0) << "Filter dimensions cannot be zero.";
  pool_stride_ = pcpp_param.pool_stride();
  pool_pad_ = pcpp_param.pool_pad();
  if (global_pooling_) {
    CHECK(pool_pad_ == 0 && pool_stride_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  conved_channels_ = this->channels_ * M_;
  pooled_channels_ = static_cast<int>(ceil(static_cast<float>(
      conved_channels_ + 2 * pool_pad_ - pool_kernel_) / pool_stride_)) + 1;
}

template <typename Dtype>
void PCPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //Pconv Reshape
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(pooled_channels_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1,this->channels_);
    bias_multiplier_shape.push_back(this->out_spatial_dim_);  
    bias_multiplier_pconv_.Reshape(bias_multiplier_shape);  
    caffe_set(bias_multiplier_pconv_.count(), Dtype(1),
        bias_multiplier_pconv_.mutable_cpu_data());  
  }
  //PCpooling Reshape
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pcpp_param().pool() ==
      PcppParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), pooled_channels_, height_out_,
        width_out_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pcpp_param().pool() ==
      PcppParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), pooled_channels_, height_,
        width_);
  }
}

template <typename Dtype>
void PCPPLayer<Dtype>::compute_output_shape() {
  this->output_shape_.clear();
  this->output_shape_.push_back(height_out_);
  this->output_shape_.push_back(width_out_);
}

template <typename Dtype>
void im2col_cpu_pconv(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  const Dtype* data_im_ori = data_im;
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
    for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
      for (int channel = channels; channel--; data_im += channel_size) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
      data_im = data_im_ori;
    }
  }
}

// Explicit instantiation
template void im2col_cpu_pconv<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void im2col_cpu_pconv<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);

template <typename Dtype>
void col2im_cpu_pconv(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  Dtype* data_im_ori = data_im;
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
    for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
      for (int channel = channels; channel--; data_im += channel_size) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
      data_im = data_im_ori;
    }
  }
}

// Explicit instantiation
template void col2im_cpu_pconv<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu_pconv<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

//
template <typename Dtype>  
void PCPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
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
  caffe_set(intermediate->count(), Dtype(0),
        intermediate->mutable_cpu_data()); 
  // Pconv
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();    
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = intermediate->mutable_cpu_data();  
    for (int n = 0; n < (this->num_); ++n) {
      im2col_cpu_pconv(bottom_data + n*this->bottom_dim_, this->channels_,
          height_,width_,
          kernel_shape_data[0], kernel_shape_data[1],
          pad_data[0], pad_data[1],
          stride_data[0], stride_data[1],
          dilation_data[0], dilation_data[1], this->get_mutable_col_buffer_cpu());
      const Dtype* col_buff = this->get_col_buffer_cpu();
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_,
        N_, K_,
        (Dtype)1., weight, col_buff,
        (Dtype)0., top_data + n*this->top_dim_);
      if (this->bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(), 
          bias_multiplier_pconv_.cpu_data(), (Dtype)1., top_data + n*this->top_dim_);
      }
    }
  }
  
  // PcPooling
  const Dtype* bottom_data = intermediate->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  const int channel_offset = height_out_ * width_out_;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pcpp_param().pool()) {
  case PcppParameter_PoolMethod_MAX:
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
    for (int n = 0; n < intermediate->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_out_; ++h) {
          for (int w = 0; w < width_out_; ++w) {
            int cstart = pc * pool_stride_;
            int cend = min(cstart + pool_kernel_, conved_channels_);
            const int pool_index = pc * channel_offset + h * width_out_ + w;
            for (int c = cstart; c < cend; ++c) {
              const int index = c * channel_offset + h * width_out_ + w;
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
      bottom_data += intermediate->offset(1);
      top_data += top[0]->offset(1);
      if (use_top_mask) {
        top_mask += top[0]->offset(1);
      } else {
        mask += top[0]->offset(1);
      }
    }
    break;
  case PcppParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < intermediate->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_out_; ++h) {
          for (int w = 0; w < width_out_; ++w) {
            int cstart = pc * pool_stride_;
            int cend = min(cstart + pool_kernel_, conved_channels_);
            int pool_size = cend - cstart;
            const int pool_index = pc * channel_offset + h * width_out_ + w;
            for (int c = cstart; c < cend; ++c) {
              const int index = c * channel_offset + h * width_out_ + w;
              top_data[pool_index] += bottom_data[index];
            }
            top_data[pool_index] /= pool_size;
          }
        }
      }
      // compute offset
      bottom_data += intermediate->offset(1);
      top_data += top[0]->offset(1);
    }
    break;
  case PcppParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  delete intermediate;
}

// need modifiction
template <typename Dtype>
void PCPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  caffe_set(intermediate->count(), Dtype(0),
        intermediate->mutable_cpu_data());

  // PcPooling
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = intermediate->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(intermediate->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  const int channel_offset = height_out_ * width_out_;
  switch (this->layer_param_.pcpp_param().pool()) {
  case PcppParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_out_; ++h) {
          for (int w = 0; w < width_out_; ++w) {
            const int index = pc * channel_offset + h * width_out_ + w;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
      }
      // compute offset
      bottom_diff += intermediate->offset(1);
      top_diff += top[0]->offset(1);
      if (use_top_mask) {
        top_mask += top[0]->offset(1);
      } else {
        mask += top[0]->offset(1);
      }
    }
    break;
  case PcppParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_out_; ++h) {
          for (int w = 0; w < width_out_; ++w) {
            int cstart = pc * pool_stride_;
            int cend = min(cstart + pool_kernel_, conved_channels_);
            int pool_size = cend - cstart;
            const int top_index = pc * channel_offset + h * width_out_ + w;
            for (int c = cstart; c < cend; ++c) {
              const int bottom_index = c * channel_offset + h * width_out_ + w;
              bottom_diff[bottom_index] += top_diff[top_index] / pool_size;
            }
          }
        }
      }
      // compute offset
      bottom_diff += intermediate->offset(1);
      top_diff += top[0]->offset(1);
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }    
  
  // Pconv
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    top_diff = intermediate->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, M_, N_,
          1., top_diff + n*this->top_dim_, bias_multiplier_pconv_.cpu_data(), 1., bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        im2col_cpu_pconv(bottom_data + n*this->bottom_dim_, this->channels_,
          height_,width_,
          kernel_shape_data[0], kernel_shape_data[1],
          pad_data[0], pad_data[1],
          stride_data[0], stride_data[1],
          dilation_data[0], dilation_data[1], this->get_mutable_col_buffer_cpu());
        const Dtype* col_buff = this->get_col_buffer_cpu();
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_,
            K_, N_,
            (Dtype)1., top_diff + n * this->top_dim_, col_buff,
            (Dtype)1., weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_,
            N_, M_,
            (Dtype)1., weight, top_diff + n * this->top_dim_,
            (Dtype)0., this->get_mutable_col_buffer_cpu());
        }
        col2im_cpu_pconv(col_buff, this->channels_,
          height_,width_,
          kernel_shape_data[0], kernel_shape_data[1],
          pad_data[0], pad_data[1],
          stride_data[0], stride_data[1],
          dilation_data[0], dilation_data[1], bottom_diff + n*this->bottom_dim_);
      }
    }
  }

  delete intermediate;
}
#ifdef CPU_ONLY
STUB_GPU(PCPPLayer);
#endif

INSTANTIATE_CLASS(PCPPLayer);
REGISTER_LAYER_CLASS(PCPP);
}  // namespace caffe
