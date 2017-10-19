#include <vector>

#include "caffe/layers/pconv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
 

namespace caffe {

//
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void PierceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Pierce Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Pierce Layer takes a single blob as output.";
      
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(this->num_output_ % this->channels_, 0) <<
    "Number of output channels should be multiples of input channels.";
    
  //Set Weight & Bias Parameters
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  vector<int> weight_shape(2); 
  weight_shape[0] = this->num_output_/this->channels_;
  weight_shape[1] = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {  
    weight_shape.push_back(kernel_shape_data[i]);
  }
  vector<int> bias_shape(this->bias_term_, this->num_output_/this->channels_);
  if (this->bias_term_) {
    this->blobs_.resize(2);
  } else {
    this->blobs_.resize(1);
  }
  // Intialize the weight
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  // fill the weight
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  if (this->bias_term_) {  
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(  
          this->layer_param_.convolution_param().bias_filler()));  
      bias_filler->Fill(this->blobs_[1].get());//fill the bias  
  }  
  
  // ****Set Parameters****
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
  M_ = this->num_output_/this->channels_; // number of kernel;
  N_ = this->channels_ * height_out_ * width_out_;
  K_ = kernel_shape_data[0]*kernel_shape_data[1];
}

template <typename Dtype>
void PierceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1,this->channels_);
    bias_multiplier_shape.push_back(this->out_spatial_dim_);  
    bias_multiplier_pconv_.Reshape(bias_multiplier_shape);  
    caffe_set(bias_multiplier_pconv_.count(), Dtype(1),
        bias_multiplier_pconv_.mutable_cpu_data());  
  }
}

template <typename Dtype>
void PierceLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
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
void PierceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();    
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();  
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
}

// need modifiction
template <typename Dtype>
void PierceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
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
}
#ifdef CPU_ONLY
STUB_GPU(PierceLayer);
#endif

INSTANTIATE_CLASS(PierceLayer);
REGISTER_LAYER_CLASS(Pierce);
}  // namespace caffe
