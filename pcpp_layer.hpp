#ifndef CAFFE_PCPP_LAYER_HPP_
#define CAFFE_PCPP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class PCPPLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit PCPPLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "PCPP"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  //
  int height_;
  int width_;
  int M_;
  int N_;
  int K_;
  int height_out_;
  int width_out_;
  int conved_channels_;
  int pool_kernel_;
  int pool_stride_;
  int pool_pad_;
  int pooled_channels_;
  bool global_pooling_;
  
  //Blob Parameter;
  Blob<Dtype> bias_multiplier_pconv_;
  //Blob<Dtype> intermediate_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_PCPP_LAYER_HPP_
