## Build
```
make
```

## Usage
```
bin/conv_bench <precision> <batch_size> <num_repeats>
```

## Output (csv format)
```
w,h,c,n,k,f_w,f_h,pad_w,pad_h,stride_w,stride_h,group,flopCnt(gOps),fwd_time(us),fwd_perf(tflops),fwd_algo,bwd_time(us),bwd_perf(tflops),bwd_algo,wrw_time(us),wrw_perf(tflops),wrw_algo
17,17,128,128,128,7,1,3,0,1,1,1,8.485,2113,4.016,FwdAlgoImplicitGEMM,2034,4.172,BwdDataAlgoImplicitGEMM,3109,2.729,BwdWeightsAlgoDirect
17,17,128,128,128,1,7,0,3,1,1,1,8.485,2168,3.914,FwdAlgoImplicitGEMM,1978,4.29,BwdDataAlgoImplicitGEMM,2177,3.898,BwdWeightsAlgoDirect
17,17,128,128,192,7,1,3,0,1,1,1,12.73,2504,5.083,FwdAlgoImplicitGEMM,2928,4.347,BwdDataAlgoImplicitGEMM,3910,3.255,BwdWeightsAlgoDirect
17,17,128,128,192,1,7,0,3,1,1,1,12.73,2552,4.987,FwdAlgoImplicitGEMM,2882,4.416,BwdDataAlgoImplicitGEMM,3417,3.725,BwdWeightsAlgoDirect
```
