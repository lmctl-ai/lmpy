// Deterministic mutation fuzzing of the safetensors reader under ASan+UBSan. Every outcome
// must be a clean lmcpp::Error (or success with self-consistent tensors); never a crash,
// sanitizer report, or allocation sized by a lying header.
#include "safetensors.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
namespace fs = std::filesystem;
static std::string le64(uint64_t n){std::string s;for(int i=0;i<8;++i)s.push_back(char((n>>(8*i))&255));return s;}
static std::string file_with(const std::string& header,size_t payload){return le64(header.size())+header+std::string(payload,'\x01');}
int main(int argc,char** argv){
    unsigned iterations=argc>1?unsigned(std::atoi(argv[1])):20000;
    fs::create_directory("fuzz-work");fs::current_path("fuzz-work");
    std::mt19937_64 rng(2026);
    const std::string seeds[]={
        R"({"a":{"dtype":"F64","shape":[2,3],"data_offsets":[0,48]},"b":{"dtype":"I64","shape":[2],"data_offsets":[48,64]},"__metadata__":{"k":"v"}})",
        R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},"b":{"dtype":"U8","shape":[4],"data_offsets":[12,16]}})",
        R"({"a":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"I32","shape":[0],"data_offsets":[1,1]}})"};
    const char* inserts[]={"9","-","[","{","\"","e","0","18446744073709551615","1e400","null","\\u0000","\xff","99999999999","3037000500"};
    unsigned ok=0,rejected=0;
    for(unsigned i=0;i<iterations;++i){
        std::string header=seeds[i%3];size_t payload=64+rng()%64;
        int edits=1+int(rng()%4);
        for(int e=0;e<edits;++e){
            switch(rng()%6){
            case 0:if(!header.empty())header[rng()%header.size()]=char(rng());break;
            case 1:if(!header.empty())header.erase(rng()%header.size(),1+rng()%4);break;
            case 2:header.insert(rng()%(header.size()+1),inserts[rng()%(sizeof inserts/sizeof *inserts)]);break;
            case 3:header.resize(rng()%(header.size()+1));break;
            case 4:payload=rng()%100;break;
            default:header+=std::string(rng()%9,' ');break;
            }
        }
        std::string bytes=file_with(header,payload);
        switch(rng()%6){
        case 0:if(!bytes.empty())bytes[rng()%std::min<size_t>(bytes.size(),8)]=char(rng());break;   // header length
        case 1:bytes.resize(rng()%(bytes.size()+1));break;
        case 2:bytes.insert(0,le64(rng()%100));break;
        default:break;
        }
        {std::ofstream f("m.safetensors",std::ios::binary);f.write(bytes.data(),std::streamsize(bytes.size()));}
        auto attempt=[&](auto&& fn){try{fn();++ok;}catch(const lmcpp::Error&){++rejected;}};
        attempt([&]{auto tensors=pycli::list_tensors("m.safetensors");for(auto& t:tensors){if(t.second.shape.size()>32)std::abort();}});
        attempt([&]{auto bundle=pycli::read_bundle("m.safetensors");for(auto& t:bundle.tensors){uint64_t n=1;for(auto d:t.second.shape)n*=uint64_t(d);if(n*(t.second.dtype=="F64"||t.second.dtype=="I64"?8:t.second.dtype=="F32"||t.second.dtype=="I32"?4:1)!=t.second.bytes.size())std::abort();}});
        attempt([&]{pycli::read_tensor({"m.safetensors","a"});});
        attempt([&]{pycli::read_tensor({"m.safetensors","b"});});
    }
    // Structure-aware phase: always-valid JSON, mutated numbers. The random-byte phase above dies in the JSON
    // parser 99.95% of the time; this one must get past it and into offset/coverage/payload handling.
    {
        const char* dtypes[]={"F64","F32","I64","I32","U8","BOOL","F16","BF16","I16","U32","F8_E4M3","X9"};
        const size_t sizes[]={8,4,8,4,1,1,2,2,2,4,1,0};
        const unsigned structured=10000;unsigned parsed=0,payload_read=0;
        auto pick=[&](uint64_t n){return rng()%n;};
        for(unsigned i=0;i<structured;++i){
            struct T{std::string name,dtype;std::vector<uint64_t> shape;uint64_t begin,end;};
            std::vector<T> ts;uint64_t cursor=0;int count=1+int(pick(4));
            for(int k=0;k<count;++k){
                size_t d=pick(7);T t;t.name=std::string(1,char('a'+k));t.dtype=dtypes[d];
                int rank=1+int(pick(3));uint64_t n=1;for(int r=0;r<rank;++r){uint64_t dim=1+pick(4);t.shape.push_back(dim);n*=dim;}
                t.begin=cursor;cursor+=n*sizes[d];t.end=cursor;ts.push_back(t);
            }
            uint64_t data_size=cursor;
            if(pick(2)){
                T& t=ts[pick(ts.size())];
                switch(pick(10)){
                case 0:t.shape[pick(t.shape.size())]=0;break;
                case 1:t.shape[pick(t.shape.size())]=uint64_t(INT64_MAX);break;
                case 2:t.shape[pick(t.shape.size())]=uint64_t(1)<<32;t.shape.push_back(uint64_t(1)<<32);break;
                case 3:std::swap(t.begin,t.end);break;
                case 4:t.begin+=1+pick(3);break;
                case 5:t.end=t.end>0?t.end-1:t.end+1;break;
                case 6:t.end=data_size+1;break;
                case 7:t.dtype=dtypes[pick(12)];break;
                case 8:ts.push_back(ts[0]);break; // duplicate key
                default:t.begin=t.begin>8?t.begin-8:0;break; // overlap
                }
            }
            std::string h="{";
            for(size_t k=0;k<ts.size();++k){
                if(k)h+=",";
                h+="\""+ts[k].name+"\":{\"dtype\":\""+ts[k].dtype+"\",\"shape\":[";
                for(size_t r=0;r<ts[k].shape.size();++r){if(r)h+=",";h+=std::to_string(ts[k].shape[r]);}
                h+="],\"data_offsets\":["+std::to_string(ts[k].begin)+","+std::to_string(ts[k].end)+"]}";
            }
            h+="}";h+=std::string((8-h.size()%8)%8,' ');
            std::string bytes=file_with(h,data_size);
            {std::ofstream f("s.safetensors",std::ios::binary);f.write(bytes.data(),std::streamsize(bytes.size()));}
            try{pycli::list_tensors("s.safetensors");++parsed;}catch(const lmcpp::Error&){}
            try{auto bundle=pycli::read_bundle("s.safetensors");++payload_read;(void)bundle;}catch(const lmcpp::Error&){}
            try{pycli::read_tensor({"s.safetensors","a"});}catch(const lmcpp::Error&){}
        }
        std::fprintf(stderr,"structured fuzz: %u/%u listed, %u/%u read payloads\n",parsed,structured,payload_read,structured);
        if(payload_read<structured/10||parsed<structured/10){std::fprintf(stderr,"structure-aware mutants no longer reach the layout/payload checks\n");return 3;}
    }
    // Pathological sizes that must be rejected from the header alone.
    {std::string h=R"({"x":{"dtype":"F64","shape":[100000000000],"data_offsets":[0,800000000000]}})";std::ofstream f("big.safetensors",std::ios::binary);auto s=le64(h.size())+h;f.write(s.data(),std::streamsize(s.size()));}
    try{pycli::read_tensor({"big.safetensors","x"});std::abort();}catch(const lmcpp::Error&){}
    std::fprintf(stderr,"fuzz: %u accepted, %u rejected\n",ok,rejected);
    fs::current_path("..");fs::remove_all("fuzz-work");
    return 0;
}
