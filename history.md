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
        
