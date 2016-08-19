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
    
    ```protobuf
    package caffeine;
    message LayerParameter {
      required string name = 1;
    }
    ```


## Updated
* `common.hpp`:
    * more marco logging `CHECK/DCHECK/CUDA_CHECK`
* `blob.hpp`, `blob.cpp`:
    * add `update()`