// Deterministic mutation fuzzing of the safetensors reader under ASan+UBSan. Every outcome
// must be a clean lmcpp::Error (or success with self-consistent tensors); never a crash,
// sanitizer report, or allocation sized by a lying header.
#include "safetensors.hpp"
#include <cstdio>
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
    // Pathological sizes that must be rejected from the header alone.
    {std::string h=R"({"x":{"dtype":"F64","shape":[100000000000],"data_offsets":[0,800000000000]}})";std::ofstream f("big.safetensors",std::ios::binary);auto s=le64(h.size())+h;f.write(s.data(),std::streamsize(s.size()));}
    try{pycli::read_tensor({"big.safetensors","x"});std::abort();}catch(const lmcpp::Error&){}
    std::fprintf(stderr,"fuzz: %u accepted, %u rejected\n",ok,rejected);
    fs::current_path("..");fs::remove_all("fuzz-work");
    return 0;
}
