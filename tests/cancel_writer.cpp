#include "safetensors.hpp"
#include <sys/resource.h>
int main(int argc,char** argv){
    if(argc!=2)return 2;
    rlimit core{0,0};setrlimit(RLIMIT_CORE,&core);
    rlimit limit{1024,1024};if(setrlimit(RLIMIT_FSIZE,&limit)!=0)return 2;
    pycli::TensorBundle bundle;pycli::TensorData tensor;
    tensor.dtype="F64";tensor.shape={2000};tensor.bytes.resize(16000);
    bundle.tensors["x"]=tensor;pycli::write_bundle(argv[1],bundle);return 3;
}
