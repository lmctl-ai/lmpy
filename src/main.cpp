#include <lmcpp/lmcpp.hpp>
#include "worker.hpp"
#include "safetensors.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <cctype>
#include <set>
#include <cmath>
#include <cstring>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>
using namespace lmcpp;
namespace fs = std::filesystem;
namespace {
constexpr size_t MAX_NUMBERS=200000, MAX_BYTES=256*1024*1024, MAX_INLINE=1000, MAX_CSV_BYTES=32*1024*1024, MAX_OUTPUT_NUMBERS=2000000;
char active_dir[4096]={0};
void cleanup_signal(int sig) {
    // All per-call filenames are fixed; signal cleanup uses POSIX-safe calls only.
    if(!active_dir[0])_exit(128+sig);
    char path[4096];
    for(const char* file:{"req.json","resp.json","worker.py","a.npy","b.npy","x.npy","output.bin","out_0.bin","out_1.bin","out_2.bin","out_3.bin","out_4.bin","out_5.bin","out_6.bin","out_7.bin"}){
        size_t n=0;while(active_dir[n] && n<sizeof(path)-64){path[n]=active_dir[n];++n;}
        path[n++]='/';while(*file)path[n++]=*file++;path[n]=0;unlink(path);
    }
    if(active_dir[0])rmdir(active_dir);
    // _exit works even as namespace PID 1. lmcpp PR_SET_PDEATHSIG kills Python.
    _exit(128+sig);
}
struct TempDir {
    std::string path;
    struct sigaction previous_term{}, previous_int{};
    TempDir(){const char* base=getenv("TMPDIR");std::string name=std::string(base?base:"/tmp")+"/pycli-XXXXXX";std::vector<char> b(name.begin(),name.end());b.push_back(0);if(!mkdtemp(b.data()))throw std::runtime_error("cannot create private temp directory");path=b.data();if(path.size()>=sizeof(active_dir)-64)throw std::runtime_error("temp path too long");std::strcpy(active_dir,path.c_str());struct sigaction action{};action.sa_handler=cleanup_signal;sigemptyset(&action.sa_mask);sigaction(SIGTERM,&action,&previous_term);sigaction(SIGINT,&action,&previous_int);}
    ~TempDir(){sigaction(SIGTERM,&previous_term,nullptr);sigaction(SIGINT,&previous_int,nullptr);std::error_code ec;fs::remove_all(path,ec);active_dir[0]=0;}
    std::string file(const char* name){return path+"/"+name;}
};
[[noreturn]] void invalid(const std::string& message,const char* code="BAD_DATA") {throw Error{ErrorCode::ValidationError,message,{{"tool_code",code}}};}
std::string confined(const std::string& name,bool output=false){
    if(name.empty()||name.find('#')!=std::string::npos)invalid("invalid artifact path","BAD_REF");
    std::error_code ec;auto root=fs::canonical(fs::current_path(),ec);auto path=fs::weakly_canonical(fs::absolute(name),ec);
    auto rel=path.lexically_relative(root);
    if(ec||rel.empty()||*rel.begin()=="..")invalid("artifact path escapes cwd","PATH_OUTSIDE_CWD");
    if(output){if(fs::exists(path)||fs::is_symlink(fs::symlink_status(path)))throw Error{ErrorCode::Conflict,"output already exists",{{"path",path.string()}}};if(!fs::is_directory(path.parent_path()))invalid("output parent must exist");}
    else{
        if(!fs::exists(path))throw Error{ErrorCode::NotFound,"input file not found: "+name,{{"tool_code","ARTIFACT_NOT_FOUND"}}};
        if(!fs::is_regular_file(path)||fs::file_size(path)>MAX_BYTES)invalid("input must be a regular file <=256 MiB");
    }
    return path.string();
}
struct Array{std::vector<int64_t> shape;std::vector<double> data;std::vector<size_t> missing;std::string dtype="F64";};
double number(const Json& v){if(v.is_number())return v.get<double>();if(v.is_null()||v=="NA"||v=="NaN")return NAN;if(v=="Inf")return INFINITY;if(v=="-Inf")return -INFINITY;invalid("expected a number or NA/NaN/Inf/-Inf");}
void flatten(const Json& value,Array& a,size_t depth=0){
    if(depth>2)invalid("S0 arrays have rank <=2");
    if(value.is_array()){
        if(value.empty())invalid("empty arrays unsupported");
        if(a.shape.size()<=depth){if(!a.data.empty())invalid("mixed array ranks");a.shape.push_back(value.size());}else if(a.shape[depth]!=static_cast<int64_t>(value.size()))invalid("ragged array");
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
struct NpyHeader{std::string descr;std::vector<uint64_t> shape;};
// Closed grammar, hand-parsed (no backtracking regex on attacker-sized headers):
// {'descr': '<d>', 'fortran_order': False, 'shape': (n,) | (n, m), } padded with spaces and a newline.
NpyHeader parse_npy_header(const std::string& h){
    size_t i=0;
    auto peek=[&]{return i<h.size()?h[i]:'\0';};
    auto eat=[&](const char* literal){size_t n=std::strlen(literal);if(h.compare(i,n,literal)!=0)invalid("unsupported npy header: need C-order little-endian numeric arrays of rank 1 or 2","BAD_NPY");i+=n;};
    auto number=[&]{size_t start=i;while(peek()>='0'&&peek()<='9')++i;if(i==start||i-start>19)invalid("invalid npy shape","BAD_NPY");return std::stoull(h.substr(start,i-start));};
    NpyHeader out;eat("{'descr': '");size_t quote=h.find('\'',i);if(quote==std::string::npos||quote-i>8)invalid("invalid npy descr","BAD_NPY");out.descr=h.substr(i,quote-i);i=quote;
    eat("', 'fortran_order': False, 'shape': (");out.shape.push_back(number());eat(",");if(peek()==' ')++i;
    if(peek()>='0'&&peek()<='9'){out.shape.push_back(number());if(peek()==',')++i;if(peek()==' ')++i;}
    eat(")");if(peek()==',')++i;if(peek()==' ')++i;eat("}");while(peek()==' ')++i;eat("\n");
    if(i!=h.size())invalid("trailing bytes in npy header","BAD_NPY");return out;
}
struct DType{const char* npy;const char* name;size_t size;};
const DType kDTypes[]={{"<f8","F64",8},{"<f4","F32",4},{"<i8","I64",8},{"<i4","I32",4},{"|b1","BOOL",1},{"|u1","U8",1}};
const DType& dtype_by_name(const std::string& name){for(const auto& d:kDTypes)if(name==d.name)return d;invalid("unsupported dtype "+name,"UNSUPPORTED_DTYPE");}
// Exact conversion to binary64: integers beyond 2^53 and non-0/1 booleans are refused, never rounded.
void to_doubles(const std::string& dtype,const uint8_t* bytes,size_t count,std::vector<double>& out){
    out.resize(count);
    for(size_t k=0;k<count;++k){
        if(dtype=="F64")std::memcpy(&out[k],bytes+8*k,8);
        else if(dtype=="F32"){float v;std::memcpy(&v,bytes+4*k,4);out[k]=v;}
        else if(dtype=="I64"||dtype=="I32"){int64_t v;if(dtype=="I64")std::memcpy(&v,bytes+8*k,8);else{int32_t w;std::memcpy(&w,bytes+4*k,4);v=w;}if(v>(int64_t(1)<<53)||v<-(int64_t(1)<<53))invalid("integer magnitude exceeds 2^53 and is not exactly representable","BAD_DATA");out[k]=double(v);}
        else if(dtype=="BOOL"){if(bytes[k]>1)invalid("boolean values must be 0 or 1","BAD_DATA");out[k]=bytes[k];}
        else out[k]=bytes[k];
    }
}
Array read_npy(const std::string& path){
    std::ifstream f(path,std::ios::binary);unsigned char magic[8];f.read(reinterpret_cast<char*>(magic),8);
    if(f.gcount()!=8||std::memcmp(magic,"\x93NUMPY",6)!=0||(magic[6]!=1&&magic[6]!=2)||magic[7]!=0)invalid("invalid npy version","BAD_NPY");
    unsigned char length[4]={0};size_t lenbytes=magic[6]==1?2:4;f.read(reinterpret_cast<char*>(length),lenbytes);size_t len=0;for(size_t i=0;i<lenbytes;++i)len|=size_t(length[i])<<(8*i);
    if(!f||len<1||len>65536)invalid("invalid npy header length","BAD_NPY");std::string header(len,0);f.read(header.data(),len);if(!f)invalid("truncated npy header","BAD_NPY");
    auto parsed=parse_npy_header(header);const DType* type=nullptr;for(const auto& d:kDTypes)if(parsed.descr==d.npy)type=&d;
    if(!type)invalid("npy dtype must be one of <f8 <f4 <i8 <i4 |b1 |u1","BAD_NPY");
    Array a;size_t n=1;for(uint64_t d:parsed.shape){if(!d)invalid("empty npy array","BAD_NPY");if(d>MAX_NUMBERS||n>MAX_NUMBERS/d)invalid("npy exceeds 200000 numbers","INPUT_TOO_LARGE");n*=d;a.shape.push_back(int64_t(d));}
    if(fs::file_size(path)!=8+lenbytes+len+n*type->size)invalid("npy payload size mismatch","BAD_NPY");
    std::vector<uint8_t> raw(n*type->size);f.read(reinterpret_cast<char*>(raw.data()),raw.size());if(!f)invalid("truncated npy payload","BAD_NPY");
    a.dtype=type->name;to_doubles(a.dtype,raw.data(),n,a.data);return a;
}
Array load(const Json& value){
    Array a;
    if(value.is_string()){
        // Syntax first (a malformed ref is a validation error), then existence (NOT_FOUND, protocol 6.1).
        auto ref=value.get<std::string>();auto pos=ref.find('#');auto base=ref.substr(0,pos);auto extension=fs::path(base).extension();
        bool npy=extension==".npy"&&pos==std::string::npos,safetensors=extension==".safetensors"&&pos!=std::string::npos;
        if(!base.empty()&&!npy&&!safetensors)invalid("use .npy or .safetensors#key");
        if(safetensors&&(pos+1==ref.size()||ref.find('#',pos+1)!=std::string::npos))invalid("tensor reference must be path.safetensors#key","BAD_REF");
        auto path=confined(base);
        if(npy)return read_npy(path);
        pycli::TensorData tensor;
        try{tensor=pycli::read_tensor({path,ref.substr(pos+1)});}catch(const Error& e){throw Error{e.code==ErrorCode::NotFound?ErrorCode::NotFound:ErrorCode::ValidationError,e.message,e.data};}
        const auto& type=dtype_by_name(tensor.dtype);if(tensor.shape.empty()||tensor.shape.size()>2||tensor.bytes.size()/type.size>MAX_NUMBERS)invalid("tensor must have rank <=2 and <=200000 values");
        a.shape=tensor.shape;a.dtype=tensor.dtype;to_doubles(tensor.dtype,tensor.bytes.data(),tensor.bytes.size()/type.size,a.data);
    }else if(value.is_array())flatten(value,a);else if(value.is_object()&&value.size()==2&&value.contains("shape")&&value.contains("data")){
        if(!value["shape"].is_array()||value["shape"].empty()||value["shape"].size()>2||!value["data"].is_array())invalid("invalid inline shape/data");
        flatten(value["data"],a);size_t count=1;auto nested=a.shape;a.shape.clear();for(const auto& d:value["shape"]){if(!d.is_number_integer()||d<=0||d>1000)invalid("invalid shape");count*=d.get<size_t>();a.shape.push_back(d.get<int64_t>());}if(count!=a.data.size())invalid("shape/data mismatch");
        // Flat data is row-major by definition; nested data must already have exactly the declared shape (no silent reshaping).
        if(nested.size()>1&&nested!=a.shape)invalid("nested data shape does not match the declared shape; flatten it or fix the shape");
    }else invalid("expected inline array or artifact ref");return a;
}
Json jnum(double v){if(std::isnan(v))return "NaN";if(std::isinf(v))return v>0?"Inf":"-Inf";return v;}
Json jnums(const std::vector<double>& v,size_t limit){Json out=Json::array();for(size_t i=0;i<v.size()&&i<limit;++i)out.push_back(jnum(v[i]));return out;}
// Counts, extrema over non-NaN values (infinities kept, like nanmin/nanmax), and an overflow/underflow-safe
// Frobenius norm with IEEE semantics: NaN if any NaN, else Inf if any Inf.
Json summarize(const Array& a){
    size_t finite=0,nan=0,pinf=0,ninf=0;bool any=false;double lo=0,hi=0,scale=0;
    for(double v:a.data){
        if(std::isnan(v)){++nan;continue;}
        if(!any){lo=hi=v;any=true;}else{lo=std::min(lo,v);hi=std::max(hi,v);}
        if(std::isinf(v)){v>0?++pinf:++ninf;continue;}
        ++finite;scale=std::max(scale,std::fabs(v));
    }
    double norm;
    if(nan)norm=NAN;else if(pinf||ninf)norm=INFINITY;else if(scale==0)norm=0;
    else{long double sum=0;for(double v:a.data){long double q=static_cast<long double>(v)/static_cast<long double>(scale);sum+=q*q;}norm=static_cast<double>(static_cast<long double>(scale)*sqrtl(sum));}
    return {{"finite_count",finite},{"nan_count",nan},{"posinf_count",pinf},{"neginf_count",ninf},{"min",any?jnum(lo):Json("NA")},{"max",any?jnum(hi):Json("NA")},{"norm",jnum(norm)}};
}
// Strict decimal grammar; NA/NaN/Inf/-Inf are the only words. Overflow and underflow-to-zero are errors, denormals are exact.
double csv_number(const std::string& t){
    if(t=="NaN")return NAN;if(t=="Inf")return INFINITY;if(t=="-Inf")return -INFINITY;
    size_t i=0,n=t.size();if(i<n&&(t[i]=='+'||t[i]=='-'))++i;
    size_t digits=0;bool nonzero=false;
    while(i<n&&t[i]>='0'&&t[i]<='9'){nonzero|=t[i]!='0';++i;++digits;}
    if(i<n&&t[i]=='.'){++i;while(i<n&&t[i]>='0'&&t[i]<='9'){nonzero|=t[i]!='0';++i;++digits;}}
    if(!digits)invalid("invalid number in CSV: "+t.substr(0,32),"BAD_DATA");
    if(i<n&&(t[i]=='e'||t[i]=='E')){++i;if(i<n&&(t[i]=='+'||t[i]=='-'))++i;size_t e=0;while(i<n&&t[i]>='0'&&t[i]<='9'){++i;++e;}if(!e)invalid("invalid exponent in CSV: "+t.substr(0,32),"BAD_DATA");}
    if(i!=n)invalid("invalid number in CSV: "+t.substr(0,32),"BAD_DATA");
    errno=0;double v=std::strtod(t.c_str(),nullptr);
    if(std::isinf(v))invalid("CSV number overflows binary64: "+t.substr(0,32),"BAD_DATA");
    if(v==0&&nonzero)invalid("CSV number underflows binary64: "+t.substr(0,32),"BAD_DATA");
    return v;
}
[[noreturn]] void bad_csv(const std::string& message){invalid(message,"BAD_CSV");}
Array parse_csv(const std::string& text,bool header,size_t limit){
    if(text.find('\0')!=std::string::npos)bad_csv("CSV contains a NUL byte");
    size_t start=text.compare(0,3,"\xEF\xBB\xBF")==0?3:0;
    Array a;size_t rows=0,cols=0,line_no=0,header_cols=0;
    while(start<text.size()){
        size_t end=text.find('\n',start);bool last=end==std::string::npos;if(last)end=text.size();
        std::string line=text.substr(start,end-start);start=end+1;++line_no;
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        if(line.find('\r')!=std::string::npos||line.find('"')!=std::string::npos||line.find('\'')!=std::string::npos)bad_csv("CSV line "+std::to_string(line_no)+" contains quote or stray CR characters");
        if(line.empty())bad_csv("blank CSV line "+std::to_string(line_no)+"; use NA for a missing single-column value");
        if(size_t(std::count(line.begin(),line.end(),','))+1+a.data.size()>limit)invalid("CSV exceeds "+std::to_string(limit)+" numbers","INPUT_TOO_LARGE");
        std::vector<std::string> fields;size_t from=0;for(;;){size_t comma=line.find(',',from);fields.push_back(line.substr(from,comma==std::string::npos?std::string::npos:comma-from));if(comma==std::string::npos)break;from=comma+1;}
        if(header&&line_no==1){header_cols=fields.size();continue;}
        if(!cols)cols=fields.size();else if(fields.size()!=cols)bad_csv("CSV line "+std::to_string(line_no)+" has "+std::to_string(fields.size())+" fields, expected "+std::to_string(cols));
        if(a.data.size()+fields.size()>limit)invalid("CSV exceeds "+std::to_string(limit)+" numbers","INPUT_TOO_LARGE");
        for(const auto& field:fields){if(field.empty()||field=="NA"){a.missing.push_back(a.data.size());a.data.push_back(NAN);}else a.data.push_back(csv_number(field));}
        ++rows;
    }
    if(!rows)bad_csv("CSV has no data rows");
    if(header&&header_cols!=cols)bad_csv("CSV header has "+std::to_string(header_cols)+" fields, data has "+std::to_string(cols));
    a.shape.push_back(int64_t(rows));if(cols>1)a.shape.push_back(int64_t(cols));
    return a;
}
std::string read_text_file(const std::string& name,size_t limit){
    auto path=confined(name);if(fs::file_size(path)>limit)invalid("file exceeds "+std::to_string(limit>>20)+" MiB","INPUT_TOO_LARGE");
    std::ifstream f(path,std::ios::binary);std::string text(fs::file_size(path),'\0');f.read(text.data(),text.size());if(!f)invalid("cannot read file");return text;
}
// Same reference grammar for every op: inline array/envelope, `file.npy`, or `file.safetensors#key`.
Array import_source(const Params& p){
    const auto& data=p["data"];std::string format=p["format"];
    if(format=="json"){if(!data.is_array()&&!data.is_object())invalid("format json needs an array or {shape,data}");return load(data);}
    if(!data.is_string())invalid("format "+format+" needs a string");
    std::string text=data;
    if(format=="csv"){if(text.size()>MAX_CSV_BYTES)invalid("inline CSV too large","INPUT_TOO_LARGE");return parse_csv(text,p["header"],MAX_INLINE);}
    if(format=="csv_file"){if(fs::path(text).extension()!=".csv")invalid("csv_file must end in .csv","BAD_REF");return parse_csv(read_text_file(text,MAX_CSV_BYTES),p["header"],MAX_NUMBERS);}
    if(format=="npy"){if(fs::path(text).extension()!=".npy"||text.find('#')!=std::string::npos)invalid("npy data must be a path ending in .npy","BAD_REF");return load(data);}
    if(text.find('#')==std::string::npos||fs::path(text.substr(0,text.find('#'))).extension()!=".safetensors")invalid("safetensors data must be path.safetensors#key","BAD_REF");
    return load(data);
}
void check_key(const std::string& key){
    bool ok=!key.empty()&&key.size()<=64&&(std::isalpha(static_cast<unsigned char>(key[0]))||key[0]=='_');
    for(char c:key)ok=ok&&(std::isalnum(static_cast<unsigned char>(c))||c=='_'||c=='.'||c=='-');
    if(!ok||key.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")!=std::string::npos)invalid("key must match [A-Za-z_][A-Za-z0-9_.-]{0,63}","BAD_KEY");
}
std::string output_path(const Params& p){auto out=confined(p["out"],true);if(fs::path(out).extension()!=".safetensors")invalid("out must end in .safetensors","BAD_REF");return out;}
Json array_import(const Params& p){
    std::string key=p["key"];check_key(key);if(key=="__metadata__")invalid("reserved key","BAD_KEY");
    auto out=output_path(p);auto a=import_source(p);
    if(p["na"]=="error"&&!a.missing.empty())invalid("input contains "+std::to_string(a.missing.size())+" missing (NA/null/empty) values; use na=nan to store NaN","NA_NOT_ALLOWED");
    pycli::TensorData tensor;tensor.dtype="F64";tensor.shape=a.shape;tensor.bytes.resize(a.data.size()*8);std::memcpy(tensor.bytes.data(),a.data.data(),tensor.bytes.size());
    pycli::TensorBundle bundle;bundle.tensors[key]=tensor;pycli::write_bundle(out,bundle);
    Json result=pycli::descriptor({out,key},tensor);auto summary=summarize(a);
    for(auto it=summary.begin();it!=summary.end();++it)result[it.key()]=it.value();
    result["source_dtype"]=a.dtype;result["na_count"]=a.missing.size();return result;
}
Json array_inspect(const Params& p){
    const auto& ref=p["ref"];int64_t limit=p["limit"];if(limit<1||limit>int64_t(MAX_INLINE))invalid("limit must be 1..1000");
    if(ref.is_string()){
        std::string text=ref;auto hash=text.find('#');
        if(hash==std::string::npos&&fs::path(text).extension()==".safetensors"){
            std::map<std::string,pycli::TensorInfo> infos;try{infos=pycli::list_tensors(confined(text));}catch(const Error& e){if(e.code==ErrorCode::ValidationError||e.code==ErrorCode::NotFound)throw;throw Error{ErrorCode::ValidationError,e.message,e.data};}
            if(infos.size()>MAX_INLINE)invalid("more than 1000 tensors","TOO_MANY_TENSORS");
            Json tensors=Json::object();for(const auto& entry:infos)tensors[entry.first]={{"dtype",entry.second.dtype},{"shape",entry.second.shape}};
            return {{"source","safetensors"},{"tensors",tensors},{"count",infos.size()}};
        }
    }
    auto a=load(ref);Json result=summarize(a);
    result["source"]=ref.is_string()?(fs::path(ref.get<std::string>()).extension()==".npy"?"npy":"safetensors"):"inline";
    result["dtype"]=a.dtype;result["shape"]=a.shape;result["size"]=a.data.size();result["preview"]=jnums(a.data,size_t(limit));result["preview_truncated"]=a.data.size()>size_t(limit);
    return result;
}
void need_finite(const Array& a,const char* name){for(double v:a.data)if(!std::isfinite(v))invalid(std::string(name)+" must contain only finite numbers (NaN/Inf/NA rejected)");}
void need_symmetric(const Array& a){size_t n=size_t(a.shape[0]);for(size_t i=0;i<n;++i)for(size_t j=0;j<i;++j)if(a.data[i*n+j]!=a.data[j*n+i])invalid("matrix must be exactly symmetric","NOT_SYMMETRIC");}
void output_too_large(const std::string& message){throw Error{ErrorCode::CommandFailed,message,{{"tool_code","OUTPUT_TOO_LARGE"}}};}
// Stored-tensor count a decomposition will emit, checked before spawning (a full QR of a tall matrix is m*m numbers).
uint64_t decomp_numbers(const std::string& kind,uint64_t m,uint64_t n,bool full,bool pivoting){
    uint64_t k=std::min(m,n);
    if(kind=="svd")return full?m*m+k+n*n:m*k+k+k*n;
    if(kind=="qr")return (full?m*m+m*n:m*k+k*n)+(pivoting?n:0);
    if(kind=="eig")return 2*n+2*n*n;
    if(kind=="eigh")return n+n*n;
    if(kind=="cholesky")return n*n;
    return m*m+m*k+k*n;
}
pycli::TensorBundle read_outputs(TempDir& temp,Json& result){
    pycli::TensorBundle bundle;uint64_t total=0;
    const Json outputs=result.value("outputs",Json::array());
    if(!outputs.is_array()||outputs.empty()||outputs.size()>8)throw std::runtime_error("worker output list invalid");
    for(size_t i=0;i<outputs.size();++i){
        std::string key=outputs[i].at("key"),dtype=outputs[i].at("dtype");auto shape=outputs[i].at("shape").get<std::vector<int64_t>>();check_key(key);
        if((dtype!="F64"&&dtype!="I64")||shape.empty()||shape.size()>2)throw std::runtime_error("worker output descriptor invalid");
        uint64_t n=1;for(auto d:shape){if(d<=0||static_cast<uint64_t>(d)>MAX_OUTPUT_NUMBERS||n>MAX_OUTPUT_NUMBERS/static_cast<uint64_t>(d))throw std::runtime_error("worker output shape exceeds limit");n*=static_cast<uint64_t>(d);}
        total+=n;if(total>MAX_OUTPUT_NUMBERS)throw std::runtime_error("worker output exceeds limit");
        auto file=temp.file(("out_"+std::to_string(i)+".bin").c_str());if(fs::file_size(file)!=n*8)throw std::runtime_error("worker output size mismatch");
        pycli::TensorData tensor;tensor.dtype=dtype;tensor.shape=shape;tensor.bytes.resize(n*8);std::ifstream f(file,std::ios::binary);f.read(reinterpret_cast<char*>(tensor.bytes.data()),tensor.bytes.size());if(!f)throw std::runtime_error("cannot read worker output");
        if(!bundle.tensors.emplace(key,std::move(tensor)).second)throw std::runtime_error("duplicate worker output key");
    }
    result.erase("outputs");return bundle;
}
[[noreturn]] void no_artifact(const Json& result,const char* message){
    Json data={{"tool_code",result.value("status",std::string("PY_ERROR"))},{"warnings",result.value("warnings",Json::array())}};
    // Keep whatever the worker measured so a caller can tell 1e-8 from 0.9.
    for(const char* field:{"kind","driver","message","reconstruction_error","rank","residual_norm"})if(result.contains(field))data[field]=result[field];
    throw Error{ErrorCode::CommandFailed,message,data};
}
// The binding rule is <=1000 numbers inline per result: count every array element (shapes, meta and warnings aside).
size_t inline_numbers(const Json& v,const std::string& key=""){
    if(key=="meta"||key=="warnings"||key=="shape")return 0;
    size_t n=0;
    if(v.is_array()){for(const auto& item:v)n+=item.is_array()||item.is_object()?inline_numbers(item):1;}
    else if(v.is_object())for(auto it=v.begin();it!=v.end();++it)n+=inline_numbers(it.value(),it.key());
    return n;
}
Json call(const std::string& op,const Params& input){
    if(op=="array_import")return array_import(input);
    if(op=="array_inspect")return array_inspect(input);
    Params p=input;std::map<std::string,Array> arrays;
    if(op=="la_solve"||op=="la_solve_to"){
        arrays["a"]=load(p["a"]);arrays["b"]=load(p["b"]);auto& a=arrays["a"];auto& b=arrays["b"];
        if(a.shape.size()!=2||a.shape[0]!=a.shape[1]||b.shape.empty()||b.shape[0]!=a.shape[0]||a.shape[0]>447)invalid("incompatible solve shapes or order >447");
        for(const auto& entry:arrays)for(double v:entry.second.data)if(!std::isfinite(v))invalid("solve requires finite numbers");
        if(p["assume_a"]!="gen")for(size_t i=0;i<static_cast<size_t>(a.shape[0]);++i)for(size_t j=0;j<i;++j)
            if(a.data[i*a.shape[0]+j]!=a.data[j*a.shape[0]+i])invalid("sym/pos require an exactly symmetric matrix","NOT_SYMMETRIC");
        if(op=="la_solve"&&b.data.size()>1000)throw Error{ErrorCode::CommandFailed,"use la_solve_to",{{"tool_code","OUTPUT_TOO_LARGE"}}};
        if(op=="la_solve_to"){p["out"]=confined(p["out"],true);if(fs::path(p["out"].get<std::string>()).extension()!=".safetensors")invalid("out must end in .safetensors");}
    }else if(op=="la_lstsq"||op=="la_lstsq_to"){
        arrays["a"]=load(p["a"]);arrays["b"]=load(p["b"]);auto& a=arrays["a"];auto& b=arrays["b"];
        if(a.shape.size()!=2)invalid("a must be a matrix");
        if(b.shape[0]!=a.shape[0])invalid("b must have as many rows as a");
        need_finite(a,"a");need_finite(b,"b");
        if(p.contains("rcond")){double r=p["rcond"];if(!(r>=0&&r<1))invalid("rcond must lie in [0,1)");}
        uint64_t numbers=uint64_t(a.shape[1])*uint64_t(b.shape.size()==2?b.shape[1]:1);
        if(numbers>MAX_OUTPUT_NUMBERS)output_too_large("solution would exceed "+std::to_string(MAX_OUTPUT_NUMBERS)+" numbers");
        uint64_t inline_total=numbers+uint64_t(b.shape.size()==2?b.shape[1]:1)+uint64_t(std::min(a.shape[0],a.shape[1]));
        if(op=="la_lstsq"&&inline_total>MAX_INLINE)output_too_large("result would hold "+std::to_string(inline_total)+" inline numbers (x, residues, singular values), more than 1000; use la_lstsq_to");
        if(op=="la_lstsq_to")p["out"]=output_path(p);
    }else if(op=="la_props"){
        arrays["a"]=load(p["a"]);auto& a=arrays["a"];if(a.shape.size()!=2)invalid("a must be a matrix");need_finite(a,"a");
        if(!p.contains("what"))p["what"]=a.shape[0]==a.shape[1]?Json::array({"det","rank","cond","norm","trace"}):Json::array({"rank","cond","norm"});
        std::set<std::string> what;for(const auto& w:p["what"])if(!what.insert(w.get<std::string>()).second)invalid("duplicate entry in what","BAD_PARAM");
        if(what.empty())invalid("what must name at least one property","BAD_PARAM");
        bool square=a.shape[0]==a.shape[1];
        for(const char* name:{"det","slogdet","trace","inverse"})if(what.count(name)&&!square)invalid(std::string(name)+" requires a square matrix");
        if(what.count("cond")&&p["cond_p"]!="2"&&!square)invalid("cond_p other than 2 requires a square matrix");
        if(what.count("inverse")&&a.data.size()>MAX_INLINE)output_too_large("inverse has "+std::to_string(a.data.size())+" numbers, more than 1000 returned inline");
        if(p.contains("rank_tol")){double t=p["rank_tol"];if(!(t>=0))invalid("rank_tol must be >= 0");}
    }else if(op=="la_decomp"){
        arrays["a"]=load(p["a"]);auto& a=arrays["a"];if(a.shape.size()!=2)invalid("a must be a matrix");need_finite(a,"a");
        std::string kind=p["kind"];bool full=p["full"],pivoting=p["pivoting"],lower=p["lower"];bool square=a.shape[0]==a.shape[1];
        if((kind=="eig"||kind=="eigh"||kind=="cholesky")&&!square)invalid(kind+" requires a square matrix");
        if(kind=="eigh"||kind=="cholesky")need_symmetric(a);
        if(pivoting&&kind!="qr")invalid("pivoting applies to qr only","BAD_PARAM");
        if(full&&kind!="svd"&&kind!="qr")invalid("full applies to svd and qr only","BAD_PARAM");
        if(!lower&&kind!="cholesky")invalid("lower=false applies to cholesky only","BAD_PARAM");
        uint64_t numbers=decomp_numbers(kind,uint64_t(a.shape[0]),uint64_t(a.shape[1]),full,pivoting);
        if(numbers>MAX_OUTPUT_NUMBERS)output_too_large(kind+" would produce "+std::to_string(numbers)+" numbers (limit "+std::to_string(MAX_OUTPUT_NUMBERS)+"); use full=false");
        p["out"]=output_path(p);
    }else if(op=="stats_dist"){
        arrays["x"]=load(p["x"]);if(arrays["x"].shape.size()!=1||arrays["x"].data.size()>1000)invalid("stats_dist requires <=1000 vector values");
        auto params=p["params"];if(!params.is_object())invalid("params must be an object");for(auto it=params.begin();it!=params.end();++it){if(it.key()!="loc"&&it.key()!="scale"&&!(p["dist"]=="t"&&it.key()=="df"))invalid("unknown distribution parameter");if(!it.value().is_number()||!std::isfinite(it.value().get<double>()))invalid("distribution parameters must be finite");}
        if(params.contains("scale")&&params["scale"]<=0)invalid("scale must be positive");if(p["dist"]=="t"&&(!params.contains("df")||params["df"]<=0))invalid("t requires positive df");p["missing"]=arrays["x"].missing;
        if(p["fn"]=="ppf")for(double probability:arrays["x"].data)if(!std::isnan(probability)&&(probability<0||probability>1))invalid("ppf probabilities must lie in [0,1]","BAD_PROBABILITY");
    }
    TempDir temp;
    for(const auto& entry:arrays){auto file=temp.file((entry.first+".npy").c_str());write_npy(entry.second,file);p.erase(entry.first);p[entry.first+"_file"]=file;}
    if(op=="la_solve_to")p["output_file"]=temp.file("output.bin");
    Json request=p;
    if(op=="la_lstsq_to"||op=="la_decomp"){request["output_prefix"]=temp.file("out_");request.erase("out");}
    {std::ofstream f(temp.file("req.json"));f<<Json{{"op",op},{"params",request}}.dump();if(!f)throw std::runtime_error("cannot write request");}
    {std::ofstream f(temp.file("worker.py"));f<<PYCLI_WORKER_SOURCE;if(!f)throw std::runtime_error("cannot write worker");}
    std::string python="python3";
#ifdef PYCLI_DEV_INTERPRETER
    if(const char* configured=getenv("PYCLI_PYTHON"))python=configured;
#endif
    std::map<std::string,std::string> env={{"PATH","/usr/local/bin:/usr/bin"},{"HOME",temp.path},{"TMPDIR",temp.path},{"LC_ALL","C.UTF-8"},{"TZ","UTC"},{"PYTHONHASHSEED","0"},{"PYTHONNOUSERSITE","1"},{"PYTHONDONTWRITEBYTECODE","1"},{"MPLBACKEND","Agg"},{"PYCLI_LIMIT_NUMBERS",std::to_string(MAX_NUMBERS)}};
    for(const char* name:{"OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_MAX_THREADS"})env[name]="1";
    #if defined(__aarch64__)
    env["OPENBLAS_CORETYPE"]="ARMV8";
#elif defined(__x86_64__)
    env["OPENBLAS_CORETYPE"]="Haswell"; // Provisional until amd64 CI measures it.
#endif
    if(const char* core=getenv("PYCLI_BLAS_CORETYPE")){std::string c=core;if(c!="ARMV8"&&c!="Haswell")invalid("unsupported BLAS coretype");env["OPENBLAS_CORETYPE"]=c;}
    int seconds=50;if(const char* value=getenv("PYCLI_TIMEOUT_S")){try{seconds=std::max(1,std::min(50,std::stoi(value)));}catch(...){}}
    auto child=run_child({python,"-P","-s","-B","-X","faulthandler",temp.file("worker.py"),temp.file("req.json"),temp.file("resp.json")},env,std::chrono::seconds(seconds),1024*1024);
    if(child.timed_out)throw Error{ErrorCode::Timeout,"Python worker timed out",{{"tool_code","TIMEOUT"}}};
    if(child.exec_failed)throw Error{ErrorCode::ServiceUnavailable,"Python interpreter unavailable",{{"tool_code","PYTHON_NOT_FOUND"}}};
    if(child.status_unknown||child.exit_code!=0)throw Error{ErrorCode::InternalError,"Python worker crashed",{{"tool_code","TOOL_CRASH"},{"signal",child.signal},{"stderr",child.err.substr(child.err.size()>4096?child.err.size()-4096:0)}}};
    auto response=read_result_file(temp.file("resp.json"),"json");
    if(!response.value("ok",false)){auto code=response.value("tool_code",std::string("PY_ERROR"));throw Error{code=="BAD_DATA"?ErrorCode::ValidationError:ErrorCode::CommandFailed,response.value("message",std::string("worker failure")),{{"tool_code",code},{"exception",response.value("exception",std::string())},{"warnings",response.value("warnings",Json::array())}}};}
    Json result=response["result"];
    if(op=="py_info")result["worker_sha256"]=PYCLI_WORKER_SHA256;
    if(op=="la_solve_to"&&result.value("status",std::string())!="OK")throw Error{ErrorCode::CommandFailed,"solve produced no artifact",{{"tool_code",result.value("status",std::string("PY_ERROR"))},{"warnings",result.value("warnings",Json::array())}}};
    if(op=="la_solve_to"){
        pycli::TensorData tensor;tensor.dtype="F64";tensor.shape=result["shape"].get<std::vector<int64_t>>();size_t n=1;for(auto d:tensor.shape){if(d<=0||static_cast<uint64_t>(d)>MAX_NUMBERS||n>MAX_NUMBERS/d)throw std::runtime_error("worker output shape exceeds limit");n*=d;}
        auto output=temp.file("output.bin");if(fs::file_size(output)!=n*8)throw std::runtime_error("worker output size mismatch");tensor.bytes.resize(n*8);std::ifstream f(output,std::ios::binary);f.read(reinterpret_cast<char*>(tensor.bytes.data()),n*8);
        pycli::TensorBundle bundle;bundle.tensors["x"]=tensor;auto path=p["out"].get<std::string>();pycli::write_bundle(path,bundle);result.erase("shape");result["x"]=pycli::descriptor({path,"x"},tensor);
    }
    if(op=="la_lstsq_to"||op=="la_decomp"){
        std::string status=result.value("status",std::string());
        if(status!="OK"&&!(op=="la_lstsq_to"&&status=="RANK_DEFICIENT"))no_artifact(result,"no artifact was produced");
        auto bundle=read_outputs(temp,result);auto path=p["out"].get<std::string>();pycli::write_bundle(path,bundle);
        Json tensors=Json::object();
        for(const auto& entry:bundle.tensors)if(op!="la_lstsq_to"||entry.first!="x")tensors[entry.first]=pycli::descriptor({path,entry.first},entry.second);
        if(op=="la_lstsq_to")result["x"]=pycli::descriptor({path,"x"},bundle.tensors.at("x"));
        if(op=="la_decomp"||!tensors.empty())result["tensors"]=tensors;
    }
    size_t shown=inline_numbers(result);
    if(shown>MAX_INLINE)throw Error{ErrorCode::CommandFailed,"CAP exceeded: result holds "+std::to_string(shown)+" inline numbers (limit 1000); the worker must write large vectors to the artifact",{{"tool_code","OUTPUT_TOO_LARGE"}}};
    return result;
}
}
int main(int argc,char** argv){
    struct sigaction action{};action.sa_handler=cleanup_signal;sigemptyset(&action.sa_mask);
    sigaction(SIGTERM,&action,nullptr);sigaction(SIGINT,&action,nullptr);
    Operation info{"py_info",Access::Read,"Pinned Python scientific runtime, BLAS pools and determinism","object",{}};
    auto solve=[](bool write){Operation op{write?"la_solve_to":"la_solve",write?Access::Write:Access::Read,"Dense real solve; inline <=1000 numbers or F64 artifact refs; read solve reports SINGULAR/NOT_POSITIVE_DEFINITE; write solve fails if no artifact","object",{Param{"a",Type::Json,true},Param{"b",Type::Json,true},Param{"assume_a",Type::String,false,"gen",{"gen","sym","pos"}}}};if(write)op.params.emplace_back("out",Type::String,true);return op;};
    Operation dist{"stats_dist",Access::Read,"Normal or Student t pdf/cdf/ppf/sf; <=1000 inline values","object",{Param{"dist",Type::String,true,nullptr,{"norm","t"}},Param{"fn",Type::String,true,nullptr,{"pdf","cdf","ppf","sf"}},Param{"x",Type::Json,true},Param{"params",Type::Object,false,Json::object()}}};
    // S1: arrays and dense linear algebra. Array params accept inline JSON, `file.npy` or `file.safetensors#key`.
    Operation import_op{"array_import",Access::Write,"Convert inline JSON, CSV, a read-only .npy or a safetensors tensor into a new F64 safetensors artifact; fails with CONFLICT if out exists","object",{
        Param{"format",Type::String,true,nullptr,{"json","csv","csv_file","npy","safetensors"},"json: inline array or {shape,data}; csv: inline CSV text; csv_file/npy: path to .csv/.npy; safetensors: path.safetensors#key."},
        Param{"data",Type::Json,true,nullptr,{},"Source data selected by format. Inline sources hold at most 1000 numbers, files at most 200000."},
        Param{"key",Type::String,false,"x",{},"Tensor name in the output artifact: [A-Za-z_][A-Za-z0-9_.-]{0,63}."},
        Param{"header",Type::Boolean,false,false,{},"csv/csv_file only: skip the first line as a header."},
        Param{"na",Type::String,false,"nan",{"nan","error"},"Missing values (null, NA, empty CSV field): store NaN, or fail with NA_NOT_ALLOWED."},
        Param{"out",Type::String,true,nullptr,{},"New .safetensors path inside the working directory."}}};
    Operation inspect_op{"array_inspect",Access::Read,"Shape, dtype, finite/NaN/Inf counts, min, max, norm and a preview of an array (<=1000 numbers); a bare .safetensors path lists its tensors from the header only","object",{
        Param{"ref",Type::Json,true,nullptr,{},"Inline array, file.npy, file.safetensors#key, or file.safetensors to list tensors."},
        Param{"limit",Type::Integer,false,20,{},"Preview length, 1..1000."}}};
    auto lstsq=[](bool write){Operation op{write?"la_lstsq_to":"la_lstsq",write?Access::Write:Access::Read,"Dense least squares / minimum-norm solve (scipy.linalg.lstsq); statuses OK, RANK_DEFICIENT, NO_CONVERGENCE, NUMERICAL_FAILURE; read form returns at most 1000 inline numbers (x + residues + singular values), write form stores x (and large residues/singular values) as safetensors","object",{
        Param{"a",Type::Json,true,nullptr,{},"m-by-n matrix, finite."},Param{"b",Type::Json,true,nullptr,{},"Right-hand side with m rows (vector or m-by-k)."},
        Param{"rcond",Type::Number,false,nullptr,{},"Singular values below rcond*largest are treated as zero, in [0,1); default machine epsilon."},
        Param{"driver",Type::String,false,"gelsd",{"gelsd","gelsy","gelss"},"LAPACK driver."}}};if(write)op.params.emplace_back("out",Type::String,true,nullptr,std::vector<Json>{},"New .safetensors path for x.");return op;};
    Operation props{"la_props",Access::Read,"Scalar matrix properties: det, slogdet, rank, cond, norm, trace and optionally the inverse (<=1000 numbers); status OK, SINGULAR (inverse withheld) or NO_CONVERGENCE; cond is Inf for rank-deficient input","object",{
        Param{"a",Type::Json,true,nullptr,{},"Finite matrix."},
        Param{"what",Type::Array,false,nullptr,{"det","slogdet","rank","cond","norm","trace","inverse"},"Properties to compute; default det, rank, cond, norm, trace.","",false,false,false,{},Type::String},
        Param{"norm_ord",Type::String,false,"fro",{"fro","1","inf","2","nuc"},"Matrix norm order."},
        Param{"cond_p",Type::String,false,"2",{"2","1","inf","fro"},"Condition number norm; values other than 2 need a square matrix."},
        Param{"rank_tol",Type::Number,false,nullptr,{},"Rank threshold on singular values; default max(m,n)*eps*sigma_max."}}};
    Operation decomp{"la_decomp",Access::Write,"Matrix factorization written to a new safetensors artifact: svd (u,s,vt), qr (q,r[,p]), eig (w_re,w_im,vr_re,vr_im), eigh (w,v), cholesky (l or u), lu (p,l,u with A=P@L@U); statuses OK, NOT_POSITIVE_DEFINITE, NO_CONVERGENCE, NUMERICAL_FAILURE (write ops fail with the status as tool_code)","object",{
        Param{"a",Type::Json,true,nullptr,{},"Finite matrix; eigh/cholesky need exact symmetry, eig/eigh/cholesky need a square matrix."},
        Param{"kind",Type::String,true,nullptr,{"svd","qr","eig","eigh","cholesky","lu"},"Factorization."},
        Param{"out",Type::String,true,nullptr,{},"New .safetensors path."},
        Param{"full",Type::Boolean,false,false,{},"svd/qr: full_matrices / full Q instead of economic."},
        Param{"pivoting",Type::Boolean,false,false,{},"qr only: column pivoting, adds int64 tensor p with A[:,p]=Q@R."},
        Param{"lower",Type::Boolean,false,true,{},"cholesky only: false returns the upper factor u."}}};
    CliSpec spec{"pycli","0.1.0",{info,solve(false),solve(true),dist,import_op,inspect_op,lstsq(false),lstsq(true),props,decomp}};
    for(auto& op:spec.operations){std::string name=op.name;op.handler=[name](const Params& p){return call(name,p);};for(auto& param:op.params)if(param.description.empty())param.description=param.name+" input; see README for shapes and limits.";}
    return run_cli(spec,argc,argv);
}
