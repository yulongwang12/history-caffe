# [7149de5 on Sep 13, 2013](https://github.com/Yangqing/caffe/tree/7149de51ceb87e204e3a50d41cb1495caab965dd)
## New
* `Makefile`
* `src/caffeine` including
    * `common.hpp`
        * using `boost::shared_ptr`
        * using marco logging scheme (comment TODO: better logging)
    * `blob.hpp`, `blob.cpp`
        * 2 constructor (one default, one with num/channels/height/width specific)
        * `Reshape()` according to new shape
            * 
            ```cpp
            data_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
            diff_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
            ```
        * 8 data pointer (mutable/non-mutable_cpu/gpu_data/diff)
        * *private*: `check_data()`, `check_diff()`, `shared_ptr<SyncedMemory> data_/diff_`
            
    * `syncedmem.hpp`, `syncedmem.cpp`
        * 2 constructor (one default, one with size specific)
        * 4 data pointer getter (mutable/non-mutable cpu/gpu_data)
        * *private*: `to_cpu()`/`to_gpu()`, cpu_ptr/gpu_ptr, SynceHead enum, head_
            * `to_cpu()`: switch `head_`, if HEAD_AT_GPU, cudaMemcpy to cpu_ptr, change `head_ = SYNCED`
            * `cpu_data`: after `to_cpu()`, return `(const void*) cpu_ptr`, (`head_ = SYNCED`)
            * `mutable_cpu_data`: after `to_cpu()`, `head_ = HEAD_AT_CPU`, return `cpu_ptr`

# [746599a on Sep 13, 2013](https://github.com/Yangqing/caffe/tree/746599ae0d58c664cbdaed4d36358a137597fad6)
## Modifed
* move `Makefile` to `src`
* `src/caffeine/syncedmem.hpp`:
    * move `SyncedHead` enum to public, add `SyncedHead head() {return head_;}`
    
## New
* add gtest under `src`
    * add `test_syncedmem.cpp` under `src/caffeine`
* __add layer_param.proto__ under `src/caffeine/proto`
    * 
    ```protobuf
    package caffeine;
    message LayerParameter {
      required string name = 1;
    }
    ```
* __add `base.h`__ under `src/caffine`:
    * explicit constructor `Layer(const LayerParameter& param)`
    * 4 virtual function: `Setup/Forward/Predict/Backward`
        * parameter list: `vector<const Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top`
        * `Backward` has third parameter `bool propagate_down`
    * *protected*: `bool initialized_`, `LayerParameter layer_param_`, `vector<Blob<Dtype> > blobs` (stores the parameters)
* __add `neuron_layer.cpp`__ under `src/caffe`, include `caffeine/base.h`

## Updated
* `common.hpp`:
    * more marco logging `CHECK/DCHECK/CUDA_CHECK`
* `blob.hpp`, `blob.cpp`:
    * add `update()`

# [3955799 on Sep 13, 2013](https://github.com/Yangqing/caffe/tree/395579905ced2570e2914226a52ad99aee4ca7ea)
__compilable now__

# [c18d41f on Sep 13, 2013](https://github.com/Yangqing/caffe/commit/c18d41f432c3fb519fbfaa4428c4ff4155ed1a54)
__convert to glog__ logging scheme, keep `CUDA_CHECK`

# [298e7a4 on Sep 14, 2013](https://github.com/Yangqing/caffe/tree/298e7a4129590599f4ce99b01a85a75636651210)
## Updated
* `common.hpp` under `src/caffine`
    * __add singleton class Caffine__
        * 
        ```cpp
        static Caffine& Get() {
          if (!singleton_) {
            singleton_.reset(new Caffeine());
          }
          return *singleton_;
        }
        ```
        * *private*: constructor with CUBLAS_CHECK, `static shared_ptr<Caffine> singleton`, cublasHandle

# [c1b20c7 on Sep 14, 2013](https://github.com/Yangqing/caffe/tree/c1b20c7318c4426eed9f8d294428c95595705a01)
## New
* `base.cpp` under `src/caffeine`
    * for `Layer<Dtype>::Forward`, switch according to `Caffeine::mode()`, if `Caffeine::CPU`, call `Forward_cpu(bottom, top)`
* `common.cpp`, split declareration and implementation

# [4fd9c3f on Sep 15, 2013](https://github.com/Yangqing/caffe/tree/4fd9c3f0943d6af94b62d00efa6928835f13cb8e)
## Updated
* `layer_param.proto` under `src/caffeine/proto`
    * __add BlobProto
        * 
        ```protobuf
        message BlobProto {
          optional int32 num = 1 [default = 0];
          optional int32 height = 2 [default = 0];
          optional int32 width = 3 [default = 0];
          optional int32 channels = 4 [default = 0];
          repeated float data = 5;
          repeated float diff = 6;
        }
        ```
* `blob.hpp`, `blob.cpp`
    * add `FromProto(const BlobProto& proto)` and `ToProto(BlobProto* proto)`
        * 
        ```cpp
        template <typename Dtype>
        void Blob<Dtype>::FromProto(const BlobProto& proto) {
          Reshape(proto.num(), proto.channels(), proto.height(), proto.width());
          // copy data
          Dtype* data_vec = mutable_cpu_data();
          for (int i = 0; i < count_; ++i) {
            data_vec[i] = proto.data(i);
          }
          Dtype* diff_vec = mutable_cpu_diff();
          for (int i = 0; i < count_; ++i) {
            diff_vec[i] = proto.diff(i);
          }
        }
        
        template <typename Dtype>
        void Blob<Dtype>::ToProto(BlobProto* proto) {
          proto->set_num(num_);
          proto->set_channels(channels_);
          proto->set_height(height_);
          proto->set_width(width_);
          proto->clear_data();
          proto->clear_diff();
          const Dtype* data_vec = cpu_data();
          for (int i = 0; i < count_; ++i) {
            proto->add_data(data_vec[i]);
          }
          const Dtype* diff_vec = cpu_diff();
          for (int i = 0; i < count_; ++i) {
            proto->add_diff(diff_vec[i]);
          }
        }
        ```
* __change base.cpp to layer.cpp__
    * change `Forward/Backward` parameter type
        * from `vector<const Blob<Dtype>*>&` to `const vector<Blob<Dtype>*>&`

# [582fa14 on Sep 16, 2013](https://github.com/Yangqing/caffe/tree/582fa142ceb1b1ae3b9f1050b31edd243d98c279)
## New
* use `mkl.h` doing math
* `filler.hpp` under `src/caffeine`
    * `virtual void Fill(Blob<Dtype>* blob)`
* `vision_layers.hpp`, `neuron_layer.cpp` under `src/caffeine`
    * only take single blob as input and output
    * __realize ReLULayer__

# [f591631 on Sep 17, 2013](https://github.com/Yangqing/caffe/tree/f59163139908126222b480865b3e44b6b312a97f)
## Updated
* `common.hpp`, `common.cpp`
    * __add Caffeine::Phase__ (enum TRAIN/TEST)
    * __add const CAFFEINE_CUDA_NUM_THREADS=512__ for backward compatibility

* `neuron_layer.cu`
    * include cpu/gpu functions in the same file
    * fixed instantiation `template class ReLULayer<float/double>`
## Deleted
* `layer.hpp`, `layer.cpp`
    * delete `Predict` function

# [002e004 on Sep 17, 2013](https://github.com/Yangqing/caffe/tree/002e004a6b3760016f28a189c458a66e0e574852)
## New
* `dropout_layer.cu`

## Updated
* `layer_param.proto` under `src/caffeine/proto`
    * flesh out LayerParameter
        * 
        ```protobuf
        // Parameters to specify layers with inner products.
        optional int32 num_output = 3; // The number of outputs for the layer
        optional bool biasterm = 4 [default = true]; // whether to have bias terms
        optional FillerParameter weight_filler = 5; // The filler for the weight
        optional FillerParameter bias_filler = 6; // The filler for the bias
        
        optional uint32 pad = 7 [default = 0]; // The padding size
        optional uint32 kernelsize = 8; // The kernel size
        optional uint32 group = 9 [default = 1]; // The group size for group conv
        optional uint32 stride = 10 [default = 1]; // The stride
        optional string pool = 11 [default = 'max']; // The pooling method
        optional float dropout_ratio = 12 [default = 0.5]; // dropout ratio
        
        optional float alpha = 13 [default = 1.]; // for local response norm
        optional float beta = 14 [default = 0.75]; // for local response norm
        ```

