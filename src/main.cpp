#include <lmcpp/lmcpp.hpp>
#include "worker.hpp"
#include "safetensors.hpp"
#include <filesystem>
#include <fstream>
#include <regex>
#include <cmath>
#include <cstring>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>
using namespace lmcpp;
namespace fs = std::filesystem;
namespace {
constexpr size_t MAX_NUMBERS=200000, MAX_BYTES=256*1024*1024;
char active_dir[4096]={0};
void cleanup_signal(int sig) {
    // All per-call filenames are fixed; signal cleanup uses POSIX-safe calls only.
    char path[4096];
    for(const char* file:{"req.json","resp.json","worker.py","a.npy","b.npy","x.npy","output.bin"}){
        size_t n=0;while(active_dir[n] && n<sizeof(path)-64){path[n]=active_dir[n];++n;}
        path[n++]='/';while(*file)path[n++]=*file++;path[n]=0;unlink(path);
    }
    rmdir(active_dir);signal(sig,SIG_DFL);kill(getpid(),sig);
}
struct TempDir {
    std::string path;
    TempDir(){const char* base=getenv("TMPDIR");std::string name=std::string(base?base:"/tmp")+"/pycli-XXXXXX";std::vector<char> b(name.begin(),name.end());b.push_back(0);if(!mkdtemp(b.data()))throw std::runtime_error("cannot create private temp directory");path=b.data();if(path.size()>=sizeof(active_dir)-64)throw std::runtime_error("temp path too long");std::strcpy(active_dir,path.c_str());signal(SIGTERM,cleanup_signal);signal(SIGINT,cleanup_signal);}
    ~TempDir(){std::error_code ec;fs::remove_all(path,ec);active_dir[0]=0;}
    std::string file(const char* name){return path+"/"+name;}
};
[[noreturn]] void invalid(const std::string& message,const char* code="BAD_DATA") {throw Error{ErrorCode::ValidationError,message,{{"tool_code",code}}};}
std::string confined(const std::string& name,bool output=false){
    if(name.empty()||name.find('#')!=std::string::npos)invalid("invalid artifact path","BAD_REF");
    std::error_code ec;auto root=fs::canonical(fs::current_path(),ec);auto path=fs::weakly_canonical(fs::absolute(name),ec);
    auto rel=path.lexically_relative(root);
    if(ec||rel.empty()||*rel.begin()=="..")invalid("artifact path escapes cwd","PATH_OUTSIDE_CWD");
    if(output){if(fs::exists(path)||fs::is_symlink(fs::symlink_status(path)))throw Error{ErrorCode::Conflict,"output already exists",{{"path",path.string()}}};if(!fs::is_directory(path.parent_path()))invalid("output parent must exist");}
    else if(!fs::is_regular_file(path)||fs::file_size(path)>MAX_BYTES)invalid("input must be a regular file <=256 MiB");
    return path.string();
}
struct Array{std::vector<int64_t> shape;std::vector<double> data;std::vector<size_t> missing;};
double number(const Json& v){if(v.is_number())return v.get<double>();if(v.is_null()||v=="NA"||v=="NaN")return NAN;if(v=="Inf")return INFINITY;if(v=="-Inf")return -INFINITY;invalid("expected a number or NA/NaN/Inf/-Inf");}
void flatten(const Json& value,Array& a,size_t depth=0){
    if(depth>2)invalid("S0 arrays have rank <=2");
    if(value.is_array()){
        if(value.empty())invalid("empty arrays unsupported");
        if(a.shape.size()<=depth)a.shape.push_back(value.size());else if(a.shape[depth]!=static_cast<int64_t>(value.size()))invalid("ragged array");
        for(const auto& item:value)flatten(item,a,depth+1);
    }else{
        if(depth!=a.shape.size())invalid("mixed array ranks");
        if(a.data.size()>=1000)invalid("inline inputs limited to 1000 numbers","INPUT_TOO_LARGE");
        if(value.is_null()||value=="NA")a.missing.push_back(a.data.size());a.data.push_back(number(value));
    }
}
void write_npy(const Array& a,const std::string& path){
    std::string shape="(";for(auto d:a.shape)shape+=std::to_string(d)+", ";shape+=")";
    std::string h="{'descr': '<f8', 'fortran_order': False, 'shape': "+shape+", }";
    h.resize(((10+h.size()+1+63)/64)*64-10-1,' ');h+='\n';
    std::ofstream f(path,std::ios::binary);const char magic[]="\x93NUMPY\x01\x00";f.write(magic,8);char len[2]={static_cast<char>(h.size()&255),static_cast<char>((h.size()>>8)&255)};f.write(len,2);f<<h;f.write(reinterpret_cast<const char*>(a.data.data()),a.data.size()*8);if(!f)throw std::runtime_error("cannot write array");
}
Array read_npy(const std::string& path){
    std::ifstream f(path,std::ios::binary);unsigned char magic[8];f.read(reinterpret_cast<char*>(magic),8);
    if(f.gcount()!=8||std::memcmp(magic,"\x93NUMPY",6)!=0||(magic[6]!=1&&magic[6]!=2)||magic[7]!=0)invalid("invalid npy version","BAD_NPY");
    unsigned char length[4]={0};size_t lenbytes=magic[6]==1?2:4;f.read(reinterpret_cast<char*>(length),lenbytes);size_t len=0;for(size_t i=0;i<lenbytes;++i)len|=size_t(length[i])<<(8*i);
    if(!f||len<1||len>65536)invalid("invalid npy header length","BAD_NPY");std::string header(len,0);f.read(header.data(),len);if(!f)invalid("truncated npy header","BAD_NPY");
    // Closed header grammar: little-endian float64, C-order, rank one or two.
    static const std::regex grammar(R"(^\{'descr': '<f8', 'fortran_order': False, 'shape': \(([0-9]+), ?(?:([0-9]+),? ?)?\),? ?\} *\n$)");
    std::smatch match;if(!std::regex_match(header,match,grammar))invalid("S0 npy requires C-order little-endian float64, rank <=2","BAD_NPY");
    Array a;size_t n=1;for(int i=1;i<=2;++i)if(match[i].matched){uint64_t d=0;try{d=std::stoull(match[i]);}catch(...){invalid("npy shape overflow","BAD_NPY");}if(!d||d>MAX_NUMBERS||n>MAX_NUMBERS/d)invalid("npy exceeds 200000 numbers","INPUT_TOO_LARGE");n*=d;a.shape.push_back(d);}
    if(fs::file_size(path)!=8+lenbytes+len+n*8)invalid("npy payload size mismatch","BAD_NPY");a.data.resize(n);f.read(reinterpret_cast<char*>(a.data.data()),n*8);if(!f)invalid("truncated npy payload","BAD_NPY");return a;
}
Array load(const Json& value){
    Array a;
    if(value.is_string()){
        auto ref=value.get<std::string>();auto pos=ref.find('#');auto path=confined(ref.substr(0,pos));
        if(fs::path(path).extension()==".npy"&&pos==std::string::npos)return read_npy(path);
        if(fs::path(path).extension()!=".safetensors"||pos==std::string::npos)invalid("use .npy or .safetensors#key");
        pycli::TensorData tensor;
        try{tensor=pycli::read_tensor({path,ref.substr(pos+1)});}catch(const Error& e){throw Error{ErrorCode::ValidationError,e.message,e.data};}
        if(tensor.dtype!="F64"||tensor.shape.empty()||tensor.shape.size()>2||tensor.bytes.size()/8>MAX_NUMBERS)invalid("S0 tensor must be F64, rank <=2, <=200000 values");a.shape=tensor.shape;a.data.resize(tensor.bytes.size()/8);std::memcpy(a.data.data(),tensor.bytes.data(),tensor.bytes.size());
    }else if(value.is_array())flatten(value,a);else if(value.is_object()&&value.size()==2&&value.contains("shape")&&value.contains("data")){
        if(!value["shape"].is_array()||value["shape"].empty()||value["shape"].size()>2||!value["data"].is_array())invalid("invalid inline shape/data");
        flatten(value["data"],a);size_t count=1;a.shape.clear();for(const auto& d:value["shape"]){if(!d.is_number_integer()||d<=0||d>1000)invalid("invalid shape");count*=d.get<size_t>();a.shape.push_back(d.get<int64_t>());}if(count!=a.data.size())invalid("shape/data mismatch");
    }else invalid("expected inline array or artifact ref");return a;
}
Json call(const std::string& op,const Params& input){
    Params p=input;std::map<std::string,Array> arrays;
    if(op=="la_solve"||op=="la_solve_to"){
        arrays["a"]=load(p["a"]);arrays["b"]=load(p["b"]);auto& a=arrays["a"];auto& b=arrays["b"];
        if(a.shape.size()!=2||a.shape[0]!=a.shape[1]||b.shape.empty()||b.shape[0]!=a.shape[0]||a.shape[0]>512)invalid("incompatible solve shapes or order >512");
        for(const auto& entry:arrays)for(double v:entry.second.data)if(!std::isfinite(v))invalid("solve requires finite numbers");
        if(op=="la_solve"&&b.data.size()>1000)throw Error{ErrorCode::CommandFailed,"use la_solve_to",{{"tool_code","OUTPUT_TOO_LARGE"}}};
        if(op=="la_solve_to"){p["out"]=confined(p["out"],true);if(fs::path(p["out"].get<std::string>()).extension()!=".safetensors")invalid("out must end in .safetensors");}
    }else if(op=="stats_dist"){
        arrays["x"]=load(p["x"]);if(arrays["x"].shape.size()!=1||arrays["x"].data.size()>1000)invalid("stats_dist requires <=1000 vector values");
        auto params=p["params"];if(!params.is_object())invalid("params must be an object");for(auto it=params.begin();it!=params.end();++it){if(it.key()!="loc"&&it.key()!="scale"&&!(p["dist"]=="t"&&it.key()=="df"))invalid("unknown distribution parameter");if(!it.value().is_number()||!std::isfinite(it.value().get<double>()))invalid("distribution parameters must be finite");}
        if(params.contains("scale")&&params["scale"]<=0)invalid("scale must be positive");if(p["dist"]=="t"&&(!params.contains("df")||params["df"]<=0))invalid("t requires positive df");p["missing"]=arrays["x"].missing;
    }
    TempDir temp;
    for(const auto& entry:arrays){auto file=temp.file((entry.first+".npy").c_str());write_npy(entry.second,file);p.erase(entry.first);p[entry.first+"_file"]=file;}
    if(op=="la_solve_to")p["output_file"]=temp.file("output.bin");
    {std::ofstream f(temp.file("req.json"));f<<Json{{"op",op},{"params",p}}.dump();if(!f)throw std::runtime_error("cannot write request");}
    {std::ofstream f(temp.file("worker.py"));f<<PYCLI_WORKER_SOURCE;if(!f)throw std::runtime_error("cannot write worker");}
    const char* configured=getenv("PYCLI_PYTHON");std::string python=configured?configured:"python3";
    std::map<std::string,std::string> env={{"PATH","/usr/local/bin:/usr/bin"},{"HOME",temp.path},{"TMPDIR",temp.path},{"LC_ALL","C.UTF-8"},{"TZ","UTC"},{"PYTHONHASHSEED","0"},{"PYTHONNOUSERSITE","1"},{"PYTHONDONTWRITEBYTECODE","1"},{"MPLBACKEND","Agg"},{"PYCLI_LIMIT_NUMBERS",std::to_string(MAX_NUMBERS)}};
    for(const char* name:{"OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_MAX_THREADS"})env[name]="1";
    #if defined(__aarch64__)
    env["OPENBLAS_CORETYPE"]="ARMV8";
#elif defined(__x86_64__)
    env["OPENBLAS_CORETYPE"]="Haswell"; // Provisional until amd64 CI measures it.
#endif
    if(const char* core=getenv("PYCLI_BLAS_CORETYPE")){std::string c=core;if(c!="ARMV8"&&c!="Haswell")invalid("unsupported BLAS coretype");env["OPENBLAS_CORETYPE"]=c;}
    int seconds=50;if(const char* value=getenv("PYCLI_TIMEOUT_S")){try{seconds=std::max(1,std::min(50,std::stoi(value)));}catch(...){}}
    auto child=run_child({python,"-I","-B","-X","faulthandler",temp.file("worker.py"),temp.file("req.json"),temp.file("resp.json")},env,std::chrono::seconds(seconds),1024*1024);
    if(child.timed_out)throw Error{ErrorCode::Timeout,"Python worker timed out",{{"tool_code","TIMEOUT"}}};
    if(child.exec_failed)throw Error{ErrorCode::ServiceUnavailable,"Python interpreter unavailable",{{"tool_code","PYTHON_NOT_FOUND"}}};
    if(child.status_unknown||child.exit_code!=0)throw Error{ErrorCode::InternalError,"Python worker crashed",{{"tool_code","TOOL_CRASH"},{"signal",child.signal},{"stderr",child.err.substr(child.err.size()>4096?child.err.size()-4096:0)}}};
    auto response=read_result_file(temp.file("resp.json"),"json");
    if(!response.value("ok",false)){auto code=response.value("tool_code",std::string("PY_ERROR"));throw Error{code=="BAD_DATA"?ErrorCode::ValidationError:ErrorCode::CommandFailed,response.value("message",std::string("worker failure")),{{"tool_code",code},{"exception",response.value("exception",std::string())},{"warnings",response.value("warnings",Json::array())}}};}
    Json result=response["result"];
    if(op=="py_info")result["worker_sha256"]=PYCLI_WORKER_SHA256;
    if(op=="la_solve_to"&&result.value("status",std::string())=="OK"){
        pycli::TensorData tensor;tensor.dtype="F64";tensor.shape=result["shape"].get<std::vector<int64_t>>();size_t n=1;for(auto d:tensor.shape){if(d<=0||static_cast<uint64_t>(d)>MAX_NUMBERS||n>MAX_NUMBERS/d)throw std::runtime_error("worker output shape exceeds limit");n*=d;}
        auto output=temp.file("output.bin");if(fs::file_size(output)!=n*8)throw std::runtime_error("worker output size mismatch");tensor.bytes.resize(n*8);std::ifstream f(output,std::ios::binary);f.read(reinterpret_cast<char*>(tensor.bytes.data()),n*8);
        pycli::TensorBundle bundle;bundle.tensors["x"]=tensor;auto path=p["out"].get<std::string>();pycli::write_bundle(path,bundle);result.erase("shape");result["x"]=pycli::descriptor({path,"x"},tensor);
    }
    return result;
}
}
int main(int argc,char** argv){
    Operation info{"py_info",Access::Read,"Pinned Python scientific runtime, BLAS pools and determinism","object",{}};
    auto solve=[](bool write){Operation op{write?"la_solve_to":"la_solve",write?Access::Write:Access::Read,"Dense real solve; inline <=1000 numbers or F64 artifact refs; singular returns SINGULAR","object",{Param{"a",Type::Json,true},Param{"b",Type::Json,true},Param{"assume_a",Type::String,false,"gen",{"gen","sym","pos"}}}};if(write)op.params.emplace_back("out",Type::String,true);return op;};
    Operation dist{"stats_dist",Access::Read,"Normal or Student t pdf/cdf/ppf/sf; <=1000 inline values","object",{Param{"dist",Type::String,true,nullptr,{"norm","t"}},Param{"fn",Type::String,true,nullptr,{"pdf","cdf","ppf","sf"}},Param{"x",Type::Json,true},Param{"params",Type::Object,false,Json::object()}}};
    CliSpec spec{"pycli","0.1.0",{info,solve(false),solve(true),dist}};
    for(auto& op:spec.operations){std::string name=op.name;op.handler=[name](const Params& p){return call(name,p);};for(auto& param:op.params)if(param.description.empty())param.description=param.name+" input; see README for S0 shapes and limits.";}
    return run_cli(spec,argc,argv);
}
