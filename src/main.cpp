#include <lmcpp/lmcpp.hpp>
#include "provenance.hpp"
#include "worker.hpp"
#include "safetensors.hpp"
#include "expr_parser.hpp"
#include <filesystem>
#include <iostream>
#include <map>
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
#include <sys/stat.h>
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
    for(const char* file:{"req.json","resp.json","worker.py","a.npy","b.npy","c.npy","a_ub.npy","b_ub.npy","a_eq.npy","b_eq.npy","x.npy","y.npy","x_new.npy","freqs.npy","points.npy","queries.npy","vi.npy","output.bin","expr_spec.json","d_0.npy","d_1.npy","d_2.npy","d_3.npy","d_4.npy","d_5.npy","d_6.npy","d_7.npy","d_8.npy","d_9.npy","d_10.npy","d_11.npy","d_12.npy","d_13.npy","d_14.npy","d_15.npy","d_16.npy","d_17.npy","d_18.npy","d_19.npy","d_20.npy","d_21.npy","d_22.npy","d_23.npy","d_24.npy","d_25.npy","d_26.npy","d_27.npy","d_28.npy","d_29.npy","d_30.npy","d_31.npy","out_0.bin","out_1.bin","out_2.bin","out_3.bin","out_4.bin","out_5.bin","out_6.bin","out_7.bin","out_8.bin","out_9.bin","out_10.bin","out_11.bin","out_12.bin","out_13.bin","out_14.bin","out_15.bin","out_16.bin","out_17.bin","out_18.bin","out_19.bin","out_20.bin","out_21.bin","out_22.bin","out_23.bin","out_24.bin","out_25.bin","out_26.bin","out_27.bin","out_28.bin","out_29.bin","out_30.bin","out_31.bin","m_0.npy","m_1.npy","m_2.npy","m_3.npy","m_4.npy","m_5.npy","m_6.npy","m_7.npy","m_8.npy","m_9.npy","m_10.npy","m_11.npy","m_12.npy","m_13.npy","m_14.npy","m_15.npy","m_16.npy","m_17.npy","m_18.npy","m_19.npy","m_20.npy","m_21.npy","m_22.npy","m_23.npy","m_24.npy","m_25.npy","m_26.npy","m_27.npy","m_28.npy","m_29.npy","m_30.npy","m_31.npy","predict.npz","frame.npz","input.csv","table.npz","export.bin","left.npz","right.npz","ref.npy","y_true.npy","y_pred.npy","y_score.npy"}){
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
#include "frames.hpp"
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
Json file_descriptor(const std::string& path,const std::string& format){return {{"ref",path},{"path",path},{"format",format},{"key",nullptr},{"dtype",nullptr},{"shape",nullptr},{"nbytes",fs::file_size(path)}};}
void artifact_tensors(Json& result,const std::string& path,const pycli::TensorBundle& bundle){Json descriptors=Json::object();for(const auto& entry:bundle.tensors){descriptors[entry.first]=pycli::descriptor({path,entry.first},entry.second);result.erase(entry.first);}result["artifacts"]=descriptors;result["outputs_in_artifact"]=true;}
Json array_import(const Params& p){
    std::string key=p["key"];check_key(key);if(key=="__metadata__")invalid("reserved key","BAD_KEY");
    auto out=output_path(p);auto a=import_source(p);
    if(p["na"]=="error"&&!a.missing.empty())invalid("input contains "+std::to_string(a.missing.size())+" missing (NA/null/empty) values; use na=nan to store NaN","NA_NOT_ALLOWED");
    pycli::TensorData tensor;tensor.dtype="F64";tensor.shape=a.shape;tensor.bytes.resize(a.data.size()*8);std::memcpy(tensor.bytes.data(),a.data.data(),tensor.bytes.size());
    pycli::TensorBundle bundle;bundle.tensors[key]=tensor;pycli::write_bundle(out,bundle);
    Json result=summarize(a);result["artifacts"]={{key,pycli::descriptor({out,key},tensor)}};result["outputs_in_artifact"]=true;
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
            return {{"status","OK"},{"source","safetensors"},{"tensors",tensors},{"count",infos.size()}};
        }
    }
    auto a=load(ref);Json result=summarize(a);
    result["status"]="OK";result["source"]=ref.is_string()?(fs::path(ref.get<std::string>()).extension()==".npy"?"npy":"safetensors"):"inline";
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
pycli::TensorBundle read_outputs(TempDir& temp,Json& result,size_t maximum=8){
    pycli::TensorBundle bundle;uint64_t total=0;
    const Json outputs=result.value("outputs",Json::array());
    if(!outputs.is_array()||outputs.empty()||outputs.size()>maximum)throw std::runtime_error("worker output list invalid");
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
Json diagnostic_value(const Json& value,size_t& left,int depth=0){
 if(depth>12)return {{"omitted",true}};
 if(value.is_string()){std::string text=value;return text.size()>4096?Json(text.substr(0,4096)+" [truncated]"):value;}
 if(value.is_array()){if(value.size()>256||value.size()>left)return {{"omitted",true},{"count",value.size()}};left-=value.size();Json out=Json::array();for(auto& item:value)out.push_back(diagnostic_value(item,left,depth+1));return out;}
 if(value.is_object()){Json out=Json::object();for(auto it=value.begin();it!=value.end();++it)out[it.key()]=diagnostic_value(it.value(),left,depth+1);return out;}
 return value;
}
[[noreturn]] void no_artifact(const Json& result,const char* message){
 Json data={{"tool_code",result.value("status",std::string("PY_ERROR"))}};size_t elements=900,bytes=0;
 for(auto it=result.begin();it!=result.end();++it){if(it.key()=="outputs"||it.key()=="model_metadata"||it.key()=="artifacts"||it.key()=="tensors")continue;auto value=diagnostic_value(it.value(),elements);size_t count=value.dump().size();if(bytes+count>48000)value={{"omitted",true}};else bytes+=count;data[it.key()]=value;}
 throw Error{ErrorCode::CommandFailed,result.value("message",std::string(message)),data};
}
// The binding rule is <=1000 numbers inline per result: count every array element (shapes, meta and warnings aside).
size_t inline_numbers(const Json& v,const std::string& key=""){
    if(key=="meta"||key=="warnings"||key=="shape"||key=="columns")return 0;
    size_t n=0;
    if(v.is_array()){for(const auto& item:v)n+=item.is_array()||item.is_object()?inline_numbers(item):1;}
    else if(v.is_object())for(auto it=v.begin();it!=v.end();++it)n+=inline_numbers(it.value(),it.key());
    return n;
}
// ---- S3: safe expressions, optimisation, quadrature, roots. Everything caller-supplied is validated here;
// the worker only receives ASTs that this parser produced and rebuilds them from a closed function table.
const Json& expr_spec(){static const Json spec=Json::parse(PYCLI_EXPR_SPEC_JSON);return spec;}
[[noreturn]] void bad_param(const std::string& message){invalid(message,"BAD_PARAM");}
void check_variable_name(const std::string& name,const char* what="variable"){
    bool ok=!name.empty()&&name.size()<=32&&name[0]>='a'&&name[0]<='z';
    for(char c:name)ok=ok&&((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_');
    if(!ok)invalid(std::string(what)+" names must match [a-z][a-z0-9_]{0,31}","BAD_NAME");
    const auto& spec=expr_spec();bool reserved=spec["functions"].contains(name)||spec["constants"].contains(name);
    for(const auto& word:spec["reserved"])reserved=reserved||word.get<std::string>()==name;
    if(reserved)invalid(std::string(what)+" name collides with a function, constant or reserved word: "+name,"EXPR_NAME_COLLISION");
}
Json parse_expression(const std::string& text,const std::set<std::string>& names,const std::map<std::string,size_t>& dims,bool reducers,const std::string& param,std::set<std::string>* identifiers=nullptr){
    try{pycli_expr::Parser parser(text,names,reducers,dims);Json ast=parser.parse();if(identifiers)*identifiers=parser.identifiers();return ast;}
    catch(const pycli_expr::ParseFailure& e){throw Error{ErrorCode::ValidationError,"invalid expression in "+param+": "+e.code+" at byte "+std::to_string(e.position),{{"tool_code",e.code},{"position",e.position},{"near",e.near},{"param",param}}};}
}
constexpr size_t MAX_VARIABLES=64, MAX_COLUMNS=32, MAX_CONSTRAINTS=16;
constexpr uint64_t MAX_JACOBIAN_ENTRIES=20000000;  // residual rows x unknowns; the worker enforces the same cap on the real count
struct Vars{std::vector<std::pair<std::string,size_t>> layout;std::vector<double> x0;std::set<std::string> names;std::map<std::string,size_t> dims;};
Vars read_vars(const Json& x0){
    Vars v;if(!x0.is_object()||x0.empty())bad_param("x0 must be a nonempty object of start values");
    // lmcpp's Json keeps insertion order and JSON objects are unordered, so the flat layout is sorted by name explicitly.
    std::vector<std::string> keys;for(auto it=x0.begin();it!=x0.end();++it)keys.push_back(it.key());std::sort(keys.begin(),keys.end());
    for(const auto& key:keys){
        check_variable_name(key);const Json& value=x0[key];
        if(value.is_number()){v.layout.emplace_back(key,0);v.x0.push_back(value.get<double>());}
        else if(value.is_array()&&!value.empty()&&value.size()<=MAX_VARIABLES){
            for(const auto& item:value){if(!item.is_number())bad_param("vector start values must be numbers");v.x0.push_back(item.get<double>());}
            v.layout.emplace_back(key,value.size());v.dims[key]=value.size();
        }else bad_param("x0 values must be numbers or nonempty number arrays of at most 64 entries");
        v.names.insert(key);
        if(v.x0.size()>MAX_VARIABLES)bad_param("at most 64 scalar unknowns in total");
    }
    for(double d:v.x0)if(!std::isfinite(d))bad_param("x0 values must be finite");
    return v;
}
Json layout_json(const Vars& v){Json out=Json::array();for(const auto& item:v.layout)out.push_back({{"n",item.first},{"len",item.second}});return out;}
struct Columns{std::vector<std::string> names;size_t rows=0;};
// Data columns: inline arrays (<=1000 numbers) or .npy/.safetensors#key refs (<=200000), rank 1, finite, equal length.
Columns read_data(const Json* data,const Vars& v,std::map<std::string,Array>& arrays){
    Columns c;if(!data)return c;
    if(!data->is_object()||data->empty()||data->size()>MAX_COLUMNS)bad_param("data must be an object with 1..32 columns");
    size_t index=0;
    for(auto it=data->begin();it!=data->end();++it,++index){
        check_variable_name(it.key(),"data column");if(v.names.count(it.key()))invalid("data column collides with an unknown: "+it.key(),"EXPR_NAME_COLLISION");
        Array a=load(it.value());if(a.shape.size()!=1)invalid("data columns must be vectors","BAD_DATA");need_finite(a,it.key().c_str());
        if(index&&a.data.size()!=c.rows)invalid("data columns must have equal length","BAD_DATA");c.rows=a.data.size();
        c.names.push_back(it.key());arrays["d_"+std::to_string(index)]=std::move(a);
    }
    return c;
}
Json data_json(const Columns& c){Json out=Json::array();for(size_t i=0;i<c.names.size();++i)out.push_back({{"n",c.names[i]},{"key","d_"+std::to_string(i)}});return out;}
std::set<std::string> all_names(const Vars& v,const Columns& c){auto names=v.names;for(const auto& n:c.names)names.insert(n);return names;}
// bounds: {name: [lo, hi]} with null/"-Inf"/"Inf" for open ends. A scalar takes one pair; a vector takes one pair (applied to every
// component) or an array of exactly one pair per component. Returned as flat null-able arrays in layout order.
void read_bounds(const Json& bounds,const Vars& v,Json& lb,Json& ub){
    if(!bounds.is_object())bad_param("bounds must be an object");
    auto read_pair=[](const Json& pair){
        if(!pair.is_array()||pair.size()!=2)invalid("bounds need [lower, upper]","BAD_BOUNDS");
        double ends[2];
        for(int k=0;k<2;++k){
            const Json& e=pair[k];
            if(e.is_null()||(e.is_string()&&e==(k?"Inf":"-Inf")))ends[k]=k?INFINITY:-INFINITY;
            else if(e.is_number()&&std::isfinite(e.get<double>()))ends[k]=e.get<double>();
            else invalid("bound ends must be finite numbers, null, \"-Inf\" (lower) or \"Inf\" (upper)","BAD_BOUNDS");
        }
        if(!(ends[0]<ends[1]))invalid("lower bound must be strictly below the upper bound","BAD_BOUNDS");
        return std::make_pair(ends[0],ends[1]);
    };
    std::map<std::string,std::vector<std::pair<double,double>>> by_name;
    for(auto it=bounds.begin();it!=bounds.end();++it){
        if(!v.names.count(it.key()))invalid("bounds name is not an unknown: "+it.key(),"BAD_BOUNDS");
        const Json& value=it.value();auto dim=v.dims.find(it.key());
        bool per_component=value.is_array()&&!value.empty()&&value[0].is_array();
        if(per_component){
            if(dim==v.dims.end()||value.size()!=dim->second)invalid("per-component bounds for "+it.key()+" need exactly one [lower, upper] pair per component","BAD_BOUNDS");
            for(const auto& pair:value)by_name[it.key()].push_back(read_pair(pair));
        }else by_name[it.key()].assign(dim==v.dims.end()?1:dim->second,read_pair(value));
    }
    lb=Json::array();ub=Json::array();size_t at=0;
    for(const auto& item:v.layout){
        size_t count=item.second?item.second:1;auto found=by_name.find(item.first);
        for(size_t k=0;k<count;++k,++at){
            double lo=-INFINITY,hi=INFINITY;if(found!=by_name.end()){lo=found->second[k].first;hi=found->second[k].second;}
            if(v.x0[at]<lo||v.x0[at]>hi)invalid("x0 lies outside its bounds: "+item.first,"BAD_BOUNDS");
            lb.push_back(std::isinf(lo)?Json(nullptr):Json(lo));ub.push_back(std::isinf(hi)?Json(nullptr):Json(hi));
        }
    }
}
int64_t int_param(const Json& in,const char* name,int64_t lo,int64_t hi){int64_t value=in[name];if(value<lo||value>hi)bad_param(std::string(name)+" must be in "+std::to_string(lo)+".."+std::to_string(hi));return value;}
Json prepare_minimize(const Params& in,std::map<std::string,Array>& arrays){
    Vars v=read_vars(in["x0"]);Columns data=read_data(in.contains("data")?&in["data"]:nullptr,v,arrays);
    auto names=all_names(v,data);bool reducers=!data.names.empty();
    bool has_bounds=in.contains("bounds"),has_constraints=in.contains("constraints")&&!in["constraints"].empty(),has_gradient=in.contains("gradient")&&!in["gradient"].empty();
    std::string method=in.contains("method")?in["method"].get<std::string>():(has_constraints?"SLSQP":has_bounds?"L-BFGS-B":"BFGS");
    if(has_bounds&&method=="BFGS")bad_param("BFGS does not support bounds; use L-BFGS-B, SLSQP, Powell, Nelder-Mead or trust-constr");
    if(has_constraints&&method!="SLSQP"&&method!="trust-constr")bad_param("constraints need method SLSQP or trust-constr");
    if(has_gradient&&(method=="Nelder-Mead"||method=="Powell"))bad_param(method+" does not use gradients; omit gradient");
    Json request={{"method",method},{"variables",layout_json(v)},{"x0_flat",v.x0},{"data",data_json(data)}};
    request["objective_ast"]=parse_expression(in["objective"],names,v.dims,reducers,"objective");
    if(has_bounds){Json lb,ub;read_bounds(in["bounds"],v,lb,ub);request["lb"]=lb;request["ub"]=ub;}
    if(has_constraints){
        const Json& list=in["constraints"];if(list.size()>MAX_CONSTRAINTS)bad_param("at most 16 constraints");
        request["constraints"]=Json::array();
        for(size_t i=0;i<list.size();++i){
            const Json& c=list[i];
            if(!c.is_object()||c.size()!=2||!c.contains("type")||!c.contains("expr")||!c["type"].is_string()||!c["expr"].is_string()||(c["type"]!="eq"&&c["type"]!="ineq"))bad_param("each constraint is {\"type\":\"eq\"|\"ineq\",\"expr\":\"...\"} (ineq means expr >= 0)");
            request["constraints"].push_back({{"type",c["type"]},{"ast",parse_expression(c["expr"],names,v.dims,reducers,"constraints["+std::to_string(i)+"]")}});
        }
    }
    if(has_gradient){
        const Json& list=in["gradient"];
        if(list.size()!=v.x0.size())invalid("gradient needs one expression per scalar unknown ("+std::to_string(v.x0.size())+" in sorted name order, vector components in index order), got "+std::to_string(list.size()),"GRADIENT_ARITY");
        request["gradient_asts"]=Json::array();
        for(size_t i=0;i<list.size();++i)request["gradient_asts"].push_back(parse_expression(list[i],names,v.dims,reducers,"gradient["+std::to_string(i)+"]"));
    }
    if(in.contains("tol")){double t=in["tol"];if(!(t>0&&t<=1))bad_param("tol must be in (0, 1]");request["tol"]=t;}
    request["maxiter"]=int_param(in,"maxiter",1,10000);request["max_evals"]=int_param(in,"max_evals",1,2000000);request["nonfinite"]=in["nonfinite"];
    return request;
}
Json prepare_least_squares(const Params& in,std::map<std::string,Array>& arrays){
    Vars v=read_vars(in["x0"]);Columns data=read_data(in.contains("data")?&in["data"]:nullptr,v,arrays);
    auto names=all_names(v,data);std::string method=in["method"],loss=in["loss"];
    bool model=in.contains("model"),residuals=in.contains("residuals");
    if(model==residuals)bad_param("give exactly one of model or residuals");
    Json request={{"method",method},{"loss",loss},{"variables",layout_json(v)},{"x0_flat",v.x0},{"data",data_json(data)}};
    if(method=="lm"&&in.contains("bounds"))bad_param("method lm does not support bounds; use trf or dogbox");
    if(method=="lm"&&loss!="linear")bad_param("method lm supports only the linear loss");
    double f_scale=in["f_scale"];if(!(f_scale>0&&std::isfinite(f_scale)))bad_param("f_scale must be positive");request["f_scale"]=f_scale;
    request["absolute_sigma"]=in["absolute_sigma"];
    size_t m=0;
    if(model){
        if(!in.contains("response")||!std::count(data.names.begin(),data.names.end(),in["response"].get<std::string>()))bad_param("model needs response naming a data column");
        request["model_ast"]=parse_expression(in["model"],names,v.dims,false,"model");request["response"]=in["response"];m=data.rows;
        if(m<1)bad_param("model needs data");
        if(uint64_t(m)*v.x0.size()>MAX_JACOBIAN_ENTRIES)invalid("residuals x unknowns exceeds the Jacobian cap of "+std::to_string(MAX_JACOBIAN_ENTRIES)+" entries","INPUT_TOO_LARGE");
        if(in.contains("sigma")){
            const Json& sigma=in["sigma"];
            if(sigma.is_string()){
                std::string column=sigma;auto at=std::find(data.names.begin(),data.names.end(),column);
                if(at==data.names.end()||column==in["response"].get<std::string>())bad_param("sigma must name another data column or be a positive number");
                for(double d:arrays["d_"+std::to_string(at-data.names.begin())].data)if(!(d>0))invalid("sigma column must be strictly positive","BAD_DATA");
                request["sigma_key"]=column;
            }else if(sigma.is_number()&&sigma.get<double>()>0&&std::isfinite(sigma.get<double>()))request["sigma_value"]=sigma;
            else bad_param("sigma must name a data column or be a positive number");
        }
    }else{
        if(in.contains("response")||in.contains("sigma"))bad_param("response and sigma apply to model only");
        const Json& list=in["residuals"];if(list.empty()||list.size()>MAX_VARIABLES)bad_param("residuals needs 1..64 expressions");
        request["residual_asts"]=Json::array();
        // Upper bound on the residual count: an expression that touches a data column is a vector of `rows`, anything else a scalar.
        uint64_t residual_count=0;
        for(size_t i=0;i<list.size();++i){
            std::set<std::string> used;
            request["residual_asts"].push_back(parse_expression(list[i],names,v.dims,false,"residuals["+std::to_string(i)+"]",&used));
            bool vector=false;for(const auto& column:data.names)vector=vector||used.count(column);
            residual_count+=vector?data.rows:1;
        }
        if(residual_count*v.x0.size()>MAX_JACOBIAN_ENTRIES)invalid("up to "+std::to_string(residual_count)+" residuals x "+std::to_string(v.x0.size())+" unknowns exceeds the Jacobian cap of "+std::to_string(MAX_JACOBIAN_ENTRIES)+" entries; use fewer vector residuals, fewer rows or fewer unknowns","INPUT_TOO_LARGE");
    }
    if(method=="lm"&&model&&m<v.x0.size())invalid("method lm needs at least as many residuals as unknowns","BAD_DATA");
    if(in.contains("bounds")){Json lb,ub;read_bounds(in["bounds"],v,lb,ub);request["lb"]=lb;request["ub"]=ub;}
    double tol=in["tol"];if(!(tol>=2.3e-16&&tol<=0.1))bad_param("tol must be in [2.3e-16, 0.1]");request["tol"]=tol;
    request["max_nfev"]=int_param(in,"max_nfev",1,100000);request["max_evals"]=int_param(in,"max_evals",1,2000000);
    return request;
}
Json bound_value(const Json& v,const char* name){
    if(v.is_number()&&std::isfinite(v.get<double>()))return v.get<double>();
    if(v.is_string()&&(v=="Inf"||v=="-Inf"))return v;
    bad_param(std::string(name)+" must be a finite number or \"Inf\"/\"-Inf\"");
}
Json prepare_quad(const Params& in){
    Json request={{"expression_ast",parse_expression(in["f"],{"x"},{},false,"f")}};
    request["a"]=bound_value(in["a"],"a");request["b"]=bound_value(in["b"],"b");
    if(request["a"].is_string()&&request["a"]==request["b"])bad_param("a and b are the same infinity");
    double epsabs=in["epsabs"],epsrel=in["epsrel"];
    if(!(epsabs>=0&&epsabs<=1)||!(epsrel>=0&&epsrel<1)||(epsabs==0&&epsrel<1e-13))bad_param("need 0<=epsabs<=1, 0<=epsrel<1 and epsrel>=1e-13 when epsabs is 0");
    request["epsabs"]=epsabs;request["epsrel"]=epsrel;request["limit"]=int_param(in,"limit",1,1000);request["max_evals"]=int_param(in,"max_evals",1,2000000);
    if(in.contains("points")&&!in["points"].empty()){
        if(request["a"].is_string()||request["b"].is_string())bad_param("points need finite bounds");
        double lo=std::min(request["a"].get<double>(),request["b"].get<double>()),hi=std::max(request["a"].get<double>(),request["b"].get<double>());
        if(in["points"].size()>MAX_VARIABLES)bad_param("at most 64 points");
        for(const auto& point:in["points"]){double d=point.get<double>();if(!(d>lo&&d<hi))bad_param("points must lie strictly between the bounds");}
        request["points"]=in["points"];
    }
    return request;
}
Json prepare_root_system(const Params& in){
    Vars v=read_vars(in["start"]);if(!v.dims.empty())bad_param("start holds scalar unknowns only (vector unknowns are not available for root systems)");
    const Json& list=in["equations"];
    if(!list.is_array()||list.empty()||list.size()>16)bad_param("equations needs 1..16 expressions");
    if(list.size()!=v.x0.size())invalid("opt_root needs as many equations ("+std::to_string(list.size())+") as unknowns ("+std::to_string(v.x0.size())+")","UNDER_DETERMINED");
    const bool bracket=in.contains("bracket")&&!in["bracket"].is_null();
    const bool explicit_method=in.contains("method")&&!in["method"].is_null();
    std::string method=explicit_method?in["method"].get<std::string>():bracket?"brentq":"hybr";
    if(bracket&&method!="brentq")bad_param("a bracket requires method brentq (or omit method)");
    if(!bracket&&in.contains("maxiter")&&!in["maxiter"].is_null())bad_param("maxiter is brentq-only; use max_evals for hybr/lm");
    Json request={{"method",method},{"variables",layout_json(v)},{"x0_flat",v.x0},{"data",Json::array()},{"equation_asts",Json::array()}};
    for(size_t i=0;i<list.size();++i){if(!list[i].is_string())bad_param("equations are expression strings");request["equation_asts"].push_back(parse_expression(list[i],v.names,{},false,"equations["+std::to_string(i)+"]"));}
    double xtol=in["xtol"],rtol=in["residual_tol"];
    if(!(xtol>0&&xtol<1))bad_param("xtol must be in (0, 1)");if(!(rtol>0&&std::isfinite(rtol)))bad_param("residual_tol must be positive");
    request["xtol"]=xtol;request["residual_tol"]=rtol;request["maxiter"]=in.contains("maxiter")&&!in["maxiter"].is_null()?int_param(in,"maxiter",1,1000):100;request["max_evals"]=int_param(in,"max_evals",1,2000000);
    if(in.contains("bracket")&&!in["bracket"].is_null()){
        const Json& b=in["bracket"];
        if(list.size()!=1)bad_param("bracket needs exactly one equation in one unknown");
        if(!b.is_array()||b.size()!=2||!b[0].is_number()||!b[1].is_number()||!std::isfinite(b[0].get<double>())||!std::isfinite(b[1].get<double>())||!(b[0].get<double>()<b[1].get<double>()))bad_param("bracket needs finite [lower, upper] with lower < upper");
        request["bracket"]=b;request["method"]="brentq";
    }else if(method=="brentq")bad_param("method brentq needs a bracket");
    return request;
}
Json prepare_root(const Params& in){
    Json request={{"expression_ast",parse_expression(in["f"],{"x"},{},false,"f")}};
    double lower=in["lower"],upper=in["upper"];if(!(std::isfinite(lower)&&std::isfinite(upper)&&lower<upper))bad_param("need finite lower < upper");
    double xtol=in["xtol"],rtol=in["rtol"];if(!(xtol>0&&std::isfinite(xtol)))bad_param("xtol must be positive");if(!(rtol>=8.881784197001252e-16&&rtol<1))bad_param("rtol must be in [4*eps, 1)");
    request["lower"]=lower;request["upper"]=upper;request["xtol"]=xtol;request["rtol"]=rtol;request["method"]=in["method"];
    request["maxiter"]=int_param(in,"maxiter",1,1000);request["max_evals"]=int_param(in,"max_evals",1,2000000);
    return request;
}
#include "s4.hpp"
#include "s5.hpp"
#include "ml.hpp"
#include "s7.hpp"
#include "s8.hpp"
#include "s8b.hpp"
#include "s8g.hpp"
int expr_check_cli(){
    try{
        std::string raw((std::istreambuf_iterator<char>(std::cin)),std::istreambuf_iterator<char>());Json in=Json::parse(raw);
        std::set<std::string> names;for(const auto& n:in.value("names",Json::array())){if(!n.is_string())throw std::runtime_error("names must be strings");names.insert(n.get<std::string>());}
        std::map<std::string,size_t> dims;if(in.contains("dims"))for(auto it=in["dims"].begin();it!=in["dims"].end();++it)dims[it.key()]=it.value().get<size_t>();
        pycli_expr::Parser parser(in.at("expression").get<std::string>(),names,in.value("allow_reducers",false),dims);Json ast=parser.parse();
        Json ids=Json::array();for(const auto& n:parser.identifiers())ids.push_back(n);
        std::cout<<Json{{"accepted",true},{"ast",ast},{"identifiers",ids},{"sha256",PYCLI_EXPR_SPEC_SHA256}}.dump()<<"\n";
    }catch(const pycli_expr::ParseFailure& e){std::cout<<Json{{"accepted",false},{"tool_code",e.code},{"position",e.position},{"near",e.near},{"sha256",PYCLI_EXPR_SPEC_SHA256}}.dump()<<"\n";}
    catch(const std::exception& e){std::cout<<Json{{"accepted",false},{"tool_code","EXPR_SYNTAX"},{"message",e.what()},{"sha256",PYCLI_EXPR_SPEC_SHA256}}.dump()<<"\n";}
    return 0;
}
Json call(const std::string& op,const Params& input){
    if(op=="array_import")return array_import(input);
    if(op=="array_inspect")return array_inspect(input);
    Params p=input;std::map<std::string,Array> arrays;
    const bool s8=s8_op(op,input);
    const bool ml=op.rfind("ml_",0)==0&&op!="ml_metrics";std::string ml_future;
    if(ml)p=prepare_ml(op,p,arrays,ml_future);
    const bool sm=op.rfind("sm_",0)==0;const bool sm_twin=sm&&op.size()>3&&op.compare(op.size()-3,3,"_to")==0;
    bool frame=(op.rfind("frame_",0)==0&&!s8)||sm||ml;std::string table_input,table_bytes,table_output;
    if(frame){
        if(op=="frame_import"||sm||ml){
            size_t sources=p.count("columns")+p.count("records")+p.count("csv")+p.count("frame");if(sources!=1)invalid("provide exactly one of columns, records, csv, frame");
        }else if(p.contains("columns")){p["select_columns"]=p["columns"];p.erase("columns");}
        if(p.contains("columns")||p.contains("records")){
            Json names=Json::array();uint64_t rows=0,cells=0;
            if(p.contains("columns")){if(!p["columns"].is_object())invalid("columns must be a column map");for(auto it=p["columns"].begin();it!=p["columns"].end();++it){names.push_back(it.key());if(!it.value().is_array())invalid("column values must be arrays");if(names.size()==1)rows=it.value().size();if(it.value().size()!=rows)invalid("ragged columns");for(auto& v:it.value())if(v.is_object()||v.is_array())invalid("nested cells unsupported");cells+=rows;}}
            else{auto& records=p["records"];if(!records.is_array()||records.empty())invalid("records must be nonempty array");rows=records.size();if(!records[0].is_object())invalid("record must be object");for(auto it=records[0].begin();it!=records[0].end();++it)names.push_back(it.key());for(auto& r:records){if(!r.is_object()||r.size()!=names.size())invalid("ragged records");for(auto& n:names){if(!r.contains(n.get<std::string>()))invalid("ragged records");auto& v=r[n.get<std::string>()];if(v.is_array()||v.is_object())invalid("nested cells unsupported");}}cells=rows*names.size();}
            frames::names(names);if(cells>MAX_INLINE||rows>frames::ROWS)invalid("inline tables limited to 1000 cells");
        }
        if(p.contains("max_rows")){int64_t n=p["max_rows"];if(n<0||n>200000)invalid("max_rows must be 0..200000");}
        if(p.contains("csv")){table_input="input.csv";table_bytes=frames::bounded_read(confined(p["csv"]),frames::CSV_BYTES);frames::csv(table_bytes,p.value("delimiter",std::string(",")),p.value("max_rows",uint64_t(200000)));p.erase("csv");}
        if(p.contains("frame")){table_input="frame.npz";auto path=confined(p["frame"]);if(fs::path(path).extension()!=".npz")invalid("table ref requires .npz");table_bytes=frames::bounded_read(path);auto metadata=frames::npz(table_bytes);p.erase("frame");
            if(op=="frame_inspect"){uint64_t rows=metadata["n_rows"],cols=metadata["columns"].size();uint64_t head=p.value("head",uint64_t(10)),tail=p.value("tail",uint64_t(0));if((std::min(rows,head)+std::min(rows,tail))*cols>1000)invalid("preview exceeds 1000 cells");}}
        if(p.contains("where"))frames::predicate(p["where"]);
        if(p.contains("rule"))frames::rule(p["rule"]);
        if(p.contains("limit")&&(p["limit"].get<int64_t>()<0||p["limit"].get<int64_t>()>2000000))invalid("limit must be 0..2M");
        for(const char* key:{"head","tail"})if(p.contains(key)&&(p[key].get<int64_t>()<0||p[key].get<int64_t>()>1000))invalid("preview rows must be 0..1000");
        if(p.contains("aggs")){auto& a=p["aggs"];std::set<std::string> allowed={"sum","mean","min","max","count","std","var","median","nunique","first","last"};uint64_t count=0;if(!a.is_object()||a.empty())invalid("aggs must be nonempty column map");for(auto it=a.begin();it!=a.end();++it){if(!it.value().is_array()||it.value().empty())invalid("aggregation list required");std::set<std::string> seen;for(auto& v:it.value()){if(!v.is_string()||!allowed.count(v.get<std::string>())||!seen.insert(v.get<std::string>()).second)invalid("invalid or duplicate aggregation");if(++count>256)invalid("aggregation column cap");}}}
        if(p.contains("sort_by")){if(!p["sort_by"].is_array()||p["sort_by"].size()>256)invalid("invalid sort_by");for(auto& item:p["sort_by"]){if(!item.is_object()||!item.contains("col")||!item["col"].is_string()||item.size()>(item.contains("desc")?2:1)||(item.contains("desc")&&!item["desc"].is_boolean()))invalid("sort_by requires {col,desc?}");}}
        if(!sm&&!ml&&p.contains("out")){table_output=confined(p["out"],true);auto ext=fs::path(table_output).extension();if(op!="frame_export"&&ext!=".npz")invalid("table out must end in .npz");p.erase("out");}
    }
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
    }else if(op=="stats_dist"&&!s8){
        dist_eval_prepare(p,arrays,1000);
    }
    if(sm)p=prepare_sm(op,p);
    const bool s7=s7_op(op),s7_write=s7&&s7_twin(op);
    if(s7)p=prepare_s7(op,input,arrays);
    S8Extra s8x;
    if(s8)p=prepare_s8(op,input,arrays,s8x);
    const bool tensor_write=s7_write||s8x.tensor_write;
    bool s4_write=op=="opt_linprog_to"||op=="ode_solve_to";bool s4=s4_write||op=="opt_linprog"||op=="ode_solve";std::string s4_out;
    bool numeric=s4||op=="opt_minimize"||op=="opt_least_squares"||op=="num_quad"||op=="num_root"||op=="opt_root";
    if(s4)p=(op=="opt_linprog"||op=="opt_linprog_to")?prepare_linprog(input,arrays):prepare_ode(input,arrays);
    else if(numeric)p=op=="opt_minimize"?prepare_minimize(input,arrays):op=="opt_least_squares"?prepare_least_squares(input,arrays):op=="num_quad"?prepare_quad(input):op=="opt_root"?prepare_root_system(input):prepare_root(input);
    if(sm||ml||s7||s8||numeric||op=="stats_dist"){double limit=input.value("time_limit",10.);if(!std::isfinite(limit)||!(limit>0&&limit<=30))bad_param("time_limit must be >0..30 seconds");p["time_limit"]=limit;}
    TempDir temp;
    int seconds=50;if(const char* value=getenv("PYCLI_TIMEOUT_S")){try{seconds=std::max(1,std::min(50,std::stoi(value)));}catch(...){}}
    if(sm||ml||s7||s8||op=="stats_dist")p["time_budget_s"]=std::max(1,seconds-10);  // worker raises TIME_LIMIT cooperatively before the hard kill
    if(numeric){
        {std::ofstream f(temp.file("expr_spec.json"));f<<PYCLI_EXPR_SPEC_JSON;if(!f)throw std::runtime_error("cannot write expression spec");}
        p["spec_file"]=temp.file("expr_spec.json");p["time_budget_s"]=s4?Json(std::max(0.01,double(seconds)-2.0)):Json(std::max(1,seconds-10));  // worker stops cooperatively before the hard kill
    }
    if(numeric&&!s4)p["time_budget_s"]=std::min(p["time_budget_s"].get<double>(),p["time_limit"].get<double>());
    if(frame){if(!table_input.empty()){auto file=temp.file(table_input.c_str());std::ofstream f(file,std::ios::binary);f.write(table_bytes.data(),table_bytes.size());if(!f)throw std::runtime_error("cannot copy table input");p[table_input=="frame.npz"?"frame_file":"csv_file"]=file;table_bytes.clear();}if(!table_output.empty())p["output_file"]=temp.file(op=="frame_export"?"export.bin":"table.npz");}
    for(const auto& entry:arrays){auto file=temp.file((entry.first+".npy").c_str());write_npy(entry.second,file);p.erase(entry.first);p[entry.first+"_file"]=file;}
    if(s8){
        for(auto& table:s8x.tables){auto file=temp.file((table.first+".npz").c_str());std::ofstream f(file,std::ios::binary);f.write(table.second.data(),table.second.size());if(!f)throw std::runtime_error("cannot copy table input");p[table.first+"_file"]=file;table.second.clear();}
        if(s8x.needs_spec){std::ofstream f(temp.file("expr_spec.json"));f<<PYCLI_EXPR_SPEC_JSON;if(!f)throw std::runtime_error("cannot write expression spec");p["spec_file"]=temp.file("expr_spec.json");}
        if(!s8x.table_out.empty()){table_output=s8x.table_out;p["output_file"]=temp.file("table.npz");}
    }
    if(op=="la_solve_to")p["output_file"]=temp.file("output.bin");
    if(ml){
        p["output_file"]=temp.file("table.npz");p["output_prefix"]=temp.file("out_");
        if(!ml_future.empty()){std::ofstream f(temp.file("predict.npz"),std::ios::binary);f.write(ml_future.data(),ml_future.size());p["predict_file"]=temp.file("predict.npz");}
        p["lock_id"]=PYCLI_ML_LOCK_SHA256;
    }
    Json request=p;if(ml)request.erase("out");
    if(s8)for(const char* key:{"out","frame","left","right"})request.erase(key);
    if(s4){if(s4_write)s4_out=p["out"].get<std::string>();request["force_artifact"]=s4_write;}
    if(s4||sm_twin||tensor_write||op=="la_lstsq_to"||op=="la_decomp"){request["output_prefix"]=temp.file("out_");request.erase("out");}
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
    auto child=run_child({python,"-P","-s","-B","-X","faulthandler",temp.file("worker.py"),temp.file("req.json"),temp.file("resp.json")},env,std::chrono::seconds(seconds),1024*1024);
    if(child.timed_out)throw Error{ErrorCode::Timeout,"Python worker timed out",{{"tool_code","TIMEOUT"}}};
    if(child.exec_failed)throw Error{ErrorCode::ServiceUnavailable,"Python interpreter unavailable",{{"tool_code","PYTHON_NOT_FOUND"}}};
    if(child.status_unknown||child.exit_code!=0)throw Error{ErrorCode::InternalError,"Python worker crashed",{{"tool_code","TOOL_CRASH"},{"signal",child.signal},{"stderr",child.err.substr(child.err.size()>4096?child.err.size()-4096:0)}}};
    auto response=read_result_file(temp.file("resp.json"),"json");
    if(!response.value("ok",false)){auto code=response.value("tool_code",std::string("PY_ERROR"));auto kind=response.value("error_class",std::string());throw Error{(code=="BAD_DATA"||kind=="validation")?ErrorCode::ValidationError:kind=="internal"?ErrorCode::InternalError:ErrorCode::CommandFailed,response.value("message",std::string("worker failure")),{{"tool_code",code},{"exception",response.value("exception",std::string())},{"warnings",response.value("warnings",Json::array())}}};}
    Json result=response["result"];
    if(ml&&op!="ml_cv"){
        if(result.value("status",std::string())!="OK"&&!(op=="ml_fit"&&result.value("status",std::string())=="RANK_DEFICIENT"&&result.contains("model_metadata")))no_artifact(result,"ML operation produced no artifact");
        auto path=p["out"].get<std::string>();
        if(op=="ml_fit"&&result.contains("model_metadata")){
            auto bundle=read_outputs(temp,result,32);bundle.metadata=result["model_metadata"].get<std::map<std::string,std::string>>();ml_model_info(bundle,false);result.erase("model_metadata");
            if(inline_numbers(result)>MAX_INLINE||result.dump().size()>65536)output_too_large("ML summary exceeds budget");
            pycli::write_bundle(path,bundle);result["artifacts"]={{"model",file_descriptor(path,"safetensors")}};result["outputs_in_artifact"]=true;
        }else{
            auto output=temp.file("table.npz");if(!fs::is_regular_file(output))no_artifact(result,"ML operation produced no prediction artifact");frames::npz(frames::bounded_read(output));
            if(inline_numbers(result)>MAX_INLINE||result.dump().size()>65536)output_too_large("ML summary exceeds budget");
            frames::install(output,path);result["artifacts"]={{"table",file_descriptor(path,"npz")}};result["outputs_in_artifact"]=true;
        }
    }
    if(op=="py_info"){result["worker_sha256"]=PYCLI_WORKER_SHA256;result["provenance"]={{"version",PYCLI_VERSION},{"git_sha",PYCLI_GIT_SHA},{"lmcpp_ref",PYCLI_LMCPP_REF},{"lock_sha256_aarch64",PYCLI_LOCK_SHA256_AARCH64},{"lock_sha256_amd64",PYCLI_LOCK_SHA256_AMD64},{"worker_sha256",PYCLI_WORKER_SHA256}};}
    if(op=="la_solve_to"&&result.value("status",std::string())!="OK")no_artifact(result,"solve produced no artifact");
    if(op=="la_solve_to"){
        pycli::TensorData tensor;tensor.dtype="F64";tensor.shape=result["shape"].get<std::vector<int64_t>>();size_t n=1;for(auto d:tensor.shape){if(d<=0||static_cast<uint64_t>(d)>MAX_NUMBERS||n>MAX_NUMBERS/d)throw std::runtime_error("worker output shape exceeds limit");n*=d;}
        auto output=temp.file("output.bin");if(fs::file_size(output)!=n*8)throw std::runtime_error("worker output size mismatch");tensor.bytes.resize(n*8);std::ifstream f(output,std::ios::binary);f.read(reinterpret_cast<char*>(tensor.bytes.data()),n*8);
        pycli::TensorBundle bundle;bundle.tensors["x"]=tensor;auto path=p["out"].get<std::string>();pycli::write_bundle(path,bundle);result.erase("shape");artifact_tensors(result,path,bundle);
    }
    if(op=="la_lstsq_to"||op=="la_decomp"||sm_twin||tensor_write){
        std::string status=result.value("status",std::string());
        if(status!="OK"&&!((op=="la_lstsq_to"||sm_twin)&&status=="RANK_DEFICIENT"))no_artifact(result,"no artifact was produced");
        auto bundle=read_outputs(temp,result);auto path=p["out"].get<std::string>();
        if(op=="interp_eval_to"&&input["extrapolate"]=="nan"){
            const auto& x=arrays.at("x");const auto& q=arrays.at("x_new");size_t dim=input["kind"]=="rbf"?size_t(x.shape[1]):1,n=x.data.size()/dim,m=q.data.size()/dim;
            std::vector<double> lo(dim,INFINITY),hi(dim,-INFINITY);for(size_t i=0;i<n;++i)for(size_t j=0;j<dim;++j){lo[j]=std::min(lo[j],x.data[i*dim+j]);hi[j]=std::max(hi[j],x.data[i*dim+j]);}
            const auto& values=bundle.tensors.at("values");if(values.dtype!="F64"||values.shape!=std::vector<int64_t>{int64_t(m)})throw std::runtime_error("invalid interpolation output shape");
            for(size_t i=0;i<m;++i){bool outside=false;for(size_t j=0;j<dim;++j)outside=outside||q.data[i*dim+j]<lo[j]||q.data[i*dim+j]>hi[j];double v;std::memcpy(&v,values.bytes.data()+8*i,8);if(outside?!std::isnan(v):!std::isfinite(v))throw std::runtime_error("invalid interpolation extrapolation mask");}
        }
        pycli::write_bundle(path,bundle);
        artifact_tensors(result,path,bundle);
    }
    if(s4){
        if(result.contains("dual_keys")&&!s4_write){Json duals=Json::object();for(const auto& k:result["dual_keys"]){auto key=k.get<std::string>();duals[key]=result["dual_"+key];result.erase("dual_"+key);}result.erase("dual_keys");result["duals"]=duals;}
        if(result.contains("outputs")){
            if(!s4_write)output_too_large("use the _to operation for S4 artifact output");
            auto bundle=read_outputs(temp,result);
            for(auto it=result.begin();it!=result.end();){if(it.value().is_object()&&it.value().contains("tensor"))it=result.erase(it);else ++it;}
            result.erase("duals");result.erase("dual_keys");artifact_tensors(result,s4_out,bundle);
            if(inline_numbers(result)>MAX_INLINE||result.dump().size()>65536)output_too_large("S4 inline result exceeds budget");
            pycli::write_bundle(s4_out,bundle);
        }else if(s4_write)no_artifact(result,"no primal/artifact was produced");
        if(result.dump().size()>65536)output_too_large("S4 inline response exceeds 64KiB");
    }
    if((frame||s8)&&!table_output.empty()){
        if(s8&&result.value("status",std::string())!="OK")no_artifact(result,"frame operation produced no successful artifact");
        auto output=p["output_file"].get<std::string>();if(!fs::is_regular_file(output))no_artifact(result,"frame operation produced no artifact");
        if(op!="frame_export"){try{frames::npz(frames::bounded_read(output));}catch(const Error&){throw Error{ErrorCode::InternalError,"worker produced invalid table artifact",{{"tool_code","BAD_TABLE_OUTPUT"}}};}}
        size_t shown=inline_numbers(result);if(shown>MAX_INLINE)output_too_large("table summary exceeds 1000 cells");
        frames::install(output,table_output);result["artifacts"]={{op=="frame_export"?"export":"table",file_descriptor(table_output,op=="frame_export"?p["format"].get<std::string>():"npz")}};result["outputs_in_artifact"]=true;
    }
    if((frame||s8)&&result.dump().size()>65536)output_too_large("inline frame response exceeds 64KiB");
    size_t shown=inline_numbers(result);
    if(shown>MAX_INLINE)throw Error{ErrorCode::CommandFailed,"CAP exceeded: result holds "+std::to_string(shown)+" inline numbers (limit 1000); the worker must write large vectors to the artifact",{{"tool_code","OUTPUT_TOO_LARGE"}}};
    return result;
}
}
int main(int argc,char** argv){
    if(argc==2&&std::string(argv[1])=="--expr-spec-hash"){std::cout<<PYCLI_EXPR_SPEC_SHA256<<"\n";return 0;}
    if(argc==2&&std::string(argv[1])=="--expr-check")return expr_check_cli();
    struct sigaction action{};action.sa_handler=cleanup_signal;sigemptyset(&action.sa_mask);
    sigaction(SIGTERM,&action,nullptr);sigaction(SIGINT,&action,nullptr);
    Operation info{"py_info",Access::Read,"Pinned Python scientific runtime, BLAS pools and determinism, plus build provenance (version, pycli git sha, pinned lmcpp revision, SHA-256 of both wheel locks, worker sha256)","object",{}};
    auto solve=[](bool write){Operation op{write?"la_solve_to":"la_solve",write?Access::Write:Access::Read,"Dense real solve; inline <=1000 numbers or F64 artifact refs; read solve reports SINGULAR/NOT_POSITIVE_DEFINITE; write solve fails if no artifact","object",{Param{"a",Type::Json,true},Param{"b",Type::Json,true},Param{"assume_a",Type::String,false,"gen",{"gen","sym","pos"}}}};if(write)op.params.emplace_back("out",Type::String,true);return op;};
    auto stats_params=[](bool write){
        std::vector<Json> families;for(const auto& f:dist_table())families.push_back(f.first);
        std::vector<Param> params={Param{"dist",Type::String,true,nullptr,families,"Distribution family. Continuous: norm (loc, scale), t (df, loc, scale), uniform (low, high), expon (scale), gamma (shape, scale), beta (a, b), lognorm (mu, sigma of the underlying normal), chi2 (df), genextreme (c, loc, scale: scipy c = -xi of the usual GEV shape xi), genpareto (c, loc, scale), skewnorm (a, loc, scale), levy_stable (alpha in (0,2], beta in [-1,1], loc, scale; scipy S1 parameterisation; pdf/cdf are slow numerical integrations of limited accuracy, at most 200 points, and a warning says so), johnsonsu (a, b, loc, scale), nct (df, nc, loc, scale; |nc|<=1000), ncf (dfn, dfd, nc, loc, scale; nc<=1000), vonmises (kappa in (0,1e5], loc, scale; at most 200 points; scipy wraps the variates into [-pi, pi] after adding loc, so loc rotates the circle: the linear mean is not loc), truncnorm (a, b, loc, scale; a and b are STANDARDISED bounds as in scipy), weibull_min (c, loc, scale), cauchy, laplace, logistic, gumbel_r (loc, scale), pareto (b, loc, scale), rayleigh (loc, scale), f (dfn, dfd, loc, scale). Discrete (integer-valued F64; pdf means the probability mass): poisson (lam <=1e9), binom (n 1..1e9, p), zipf (a>1, loc; cdf/sf need |x| <= 1e5 and a quantile beyond 1e5 is the status OUT_OF_RANGE), betabinom (n 1..5000, a, b, loc), nbinom (n>0, p, loc), geom (p, loc). Unknown parameters are BAD_PARAM; only the parameters listed per family are accepted."},
            Param{"fn",Type::String,true,nullptr,write?std::vector<Json>{"pdf","logpdf","cdf","logcdf","sf","logsf","ppf","isf","rvs"}:std::vector<Json>{"pdf","logpdf","cdf","logcdf","sf","logsf","ppf","isf","moments","rvs"},write?"pdf/logpdf/cdf/logcdf/sf/logsf/ppf/isf at x (<=200,000 points) or rvs: seeded random variates; the result is an F64 vector artifact.":"pdf/logpdf/cdf/logcdf/sf/logsf/ppf/isf at x (<=1000 points; pdf/logpdf are the mass for discrete families; ppf/isf take probabilities in [0,1]), moments (mean, var, skew, excess kurtosis; infinite or undefined moments are reported as NA and listed in moments_undefined), or rvs: seeded random variates."},
            Param{"params",Type::Object,false,Json::object(),{},"Distribution parameters (see dist); unknown or missing required keys are BAD_PARAM; scale-like parameters must be > 0."},
            Param{"n",Type::Integer,false,nullptr,{},"rvs: sample size >=1; at most 1000 inline, at most 1,000,000 with stats_dist_to."},
            Param{"seed",Type::Integer,false,nullptr,{},"rvs: REQUIRED explicit seed, an integer in [0, 2^63). The generator is numpy PCG64(seed); the seed and generator are echoed, and the same seed gives the same stream (a shorter n is a prefix)."}};
        if(!write)params.push_back(Param{"x",Type::Json,false,nullptr,{},"Evaluation points (not allowed with rvs or moments): a vector of <=1000 numbers (inline, .npy or safetensors#key); NA entries stay NA."});
        else{params.push_back(Param{"x",Type::Json,false,nullptr,{},"Evaluation points (not allowed with rvs): a vector of <=200,000 numbers (inline, .npy or safetensors#key); NA entries become NaN in the artifact."});params.push_back(Param{"key",Type::String,false,"x",{},"Tensor name in the artifact: [A-Za-z_][A-Za-z0-9_.-]{0,63}."});params.push_back(Param{"out",Type::String,true,nullptr,{},"New .safetensors path inside the working directory (CONFLICT if it exists)."});}
        return params;};
    Operation dist{"stats_dist",Access::Read,"Distribution functions (pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf, moments; <=1000 inline values) and seeded sampling (fn rvs, PCG64, explicit required seed, <=1000 inline values) for 31 continuous and discrete families, see dist","object",stats_params(false)};
    Operation dist_to{"stats_dist_to",Access::Write,"Distribution functions at up to 200,000 points, or seeded sampling (same families, params and explicit required seed as stats_dist), written to an exclusive safetensors artifact (F64 vector under artifacts[key])","object",stats_params(true)};
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
    CliSpec spec{"pycli",PYCLI_VERSION,{info,solve(false),solve(true),dist,dist_to,import_op,inspect_op,lstsq(false),lstsq(true),props,decomp}};
    add_s7_ops(spec);
    add_s8_ops(spec);
    add_s8b_ops(spec);
    add_graph_ops(spec);


    auto common=[](){return std::vector<Param>{Param{"frame",Type::String,true,nullptr,{},"Stored-only .npz table ref inside cwd."}};};
    std::vector<Param> import_params={Param{"columns",Type::Object,false,nullptr,{},"Inline column map, <=1000 cells."},Param{"records",Type::Array,false,nullptr,{},"Inline records with identical keys, <=1000 cells.","",false,false,false,{},Type::Object},Param{"csv",Type::String,false,nullptr,{},"UTF-8 CSV path with header, <=16MiB, 200000 rows, 500000 cells; string width <=65536 bytes."},Param{"frame",Type::String,false,nullptr,{},"Existing table ref to copy."},Param{"dtypes",Type::Object,false,Json::object(),{},"Column dtype overrides: int64,float64,bool,string,datetime64[ns],category; object specs may give tz/categories/ordered."},Param{"delimiter",Type::String,false,",",{},"Single-byte CSV delimiter."},Param{"na_values",Type::Array,false,Json::array(),{},"CSV missing tokens in addition to empty cells.","",false,false,false,{},Type::String},Param{"parse_dates",Type::Array,false,Json::array(),{},"ISO datetime columns; timezone requires a dtype object override.","",false,false,false,{},Type::String},Param{"max_rows",Type::Integer,false,200000,{},"CSV maximum data rows, 0..200000."},Param{"out",Type::String,true,nullptr,{},"New .npz table artifact; CONFLICT if exists."}};
    spec.operations.push_back(Operation{"frame_import",Access::Write,"Import typed tables; inference is returned explicitly","object",import_params});
    auto inspect=common();inspect.emplace_back("head",Type::Integer,false,10,std::vector<Json>{},"Head rows; total head+tail <=1000 cells.");inspect.emplace_back("tail",Type::Integer,false,0,std::vector<Json>{},"Tail rows; total head+tail <=1000 cells.");inspect.emplace_back("describe",Type::Boolean,false,false,std::vector<Json>{},"Numeric describe statistics; combined inline cap 1000 cells.");spec.operations.push_back(Operation{"frame_inspect",Access::Read,"Shape, dtype policies, NA masks, bounded previews and statistics","object",inspect});
    auto exportp=common();exportp.emplace_back("format",Type::String,false,"csv",std::vector<Json>{"csv","json"},"Export UTF-8 CSV or columnar JSON.");exportp.emplace_back("out",Type::String,false,nullptr,std::vector<Json>{},"New output path; omit only for <=1000 cells inline. Atomic no-replace write.");spec.operations.push_back(Operation{"frame_export",Access::Write,"Export a table inline or to an exclusive file","object",exportp});
    auto select=common();select.emplace_back("columns",Type::Array,false,nullptr,std::vector<Json>{},"Selected columns in requested order.","",false,false,false,std::vector<std::string>{},Type::String);select.emplace_back("where",Type::Object,false,nullptr,std::vector<Json>{},"Predicate tree: and/or/not; col/op/value leaves with eq/ne/lt/le/gt/ge/isin/isnull/between/startswith/contains literal.");select.emplace_back("sort_by",Type::Array,false,Json::array(),std::vector<Json>{},"Stable sort list of {col,desc?}.","",false,false,false,std::vector<std::string>{},Type::Object);select.emplace_back("limit",Type::Integer,false,nullptr,std::vector<Json>{},"Maximum selected rows, 0..2M.");select.emplace_back("out",Type::String,true,nullptr,std::vector<Json>{},"New .npz table artifact.");spec.operations.push_back(Operation{"frame_select",Access::Write,"Structured filtering, column selection and stable sorting","object",select});
    for(bool resample:{false,true}){auto params=common();if(resample){params.emplace_back("time",Type::String,true,nullptr,std::vector<Json>{},"Datetime column materialized as output bins; NaT rejected.");params.emplace_back("rule",Type::String,true,nullptr,std::vector<Json>{},"1..10000 followed by min,h,D,W,MS,ME,QS,YS; no other offset syntax.");for(const char* k:{"closed","label"})params.emplace_back(k,Type::String,false,nullptr,std::vector<Json>{"left","right"},"Bin edge closure or label; pandas defaults: right for ME/W, left otherwise.");}else{params.emplace_back("keys",Type::Array,true,nullptr,std::vector<Json>{},"Group key columns; output keys are ordinary columns.","",false,false,false,std::vector<std::string>{},Type::String);params.emplace_back("dropna",Type::Boolean,false,false,std::vector<Json>{},"Drop groups whose keys are missing.");}params.emplace_back("aggs",Type::Object,true,nullptr,std::vector<Json>{},"Column -> list of sum,mean,min,max,count,std,var,median,nunique,first,last. Outputs named column__function; std/var ddof=1.");params.emplace_back("out",Type::String,true,nullptr,std::vector<Json>{},"New .npz table artifact.");spec.operations.push_back(Operation{resample?"frame_resample":"frame_groupby",Access::Write,resample?"Timezone-aware bounded resampling":"Closed-function grouped aggregates, sorted keys and observed categories","object",params});}
    // S3: caller expressions are parsed to a closed AST here; Python rebuilds them from a fixed function table.
    auto obj=[](const char* name,bool required,const char* text){return Param{name,Type::Object,required,nullptr,{},text};};
    auto num=[](const char* name,Json def,const char* text){return Param{name,Type::Number,false,std::move(def),{},text};};
    auto integer=[](const char* name,Json def,const char* text){return Param{name,Type::Integer,false,std::move(def),{},text};};
    const char* x0_text="Start values {name: number | [numbers]} (names [a-z][a-z0-9_]{0,31}; at most 64 scalars). A vector variable is used as name[0], name[1] with integer literals. Variables are ordered by name, vector components by index.";
    const char* data_text="Data columns {name: inline array (<=1000 numbers) | file.npy | file.safetensors#key}: finite vectors of equal length (<=200000), at most 32.";
    for(bool write:{false,true}){
    Operation operation{write?"opt_linprog_to":"opt_linprog",write?Access::Write:Access::Read,"Dense LP/MILP via HiGHS with row/objective scaling and original-data primal/KKT checks. Status OK/INFEASIBLE/UNBOUNDED/INFEASIBLE_OR_UNBOUNDED/MODEL_ERROR/TIME_LIMIT/NODE_LIMIT/ITERATION_LIMIT/NUMERICAL_FAILURE. LP dual signs follow the original objective. outputs_in_artifact reports descriptors for x/slack/duals; use array_inspect for previews.","object",{
        Param{"c",Type::Json,true,nullptr,{},"Finite objective vector, 1..512 variables; inline <=1000 or npy/safetensors#key. All LP inputs have finite abs <=1e15; nonzero coefficient ratios per row/objective >=1e-9."},
        Param{"a_ub",Type::Json,false,nullptr,{},"Dense inequality matrix; a_ub*x<=b_ub. Combined matrices <=200000 entries /4096 rows."},Param{"b_ub",Type::Json,false,nullptr,{},"Inequality RHS vector; required with a_ub."},
        Param{"a_eq",Type::Json,false,nullptr,{},"Dense equality matrix; a_eq*x=b_eq."},Param{"b_eq",Type::Json,false,nullptr,{},"Equality RHS vector; required with a_eq."},
        Param{"bounds",Type::Json,false,nullptr,{},"Per-variable [lower,upper], equal bounds allowed. Default [0,Inf]; null or signed -Inf/Inf tokens for open bounds; finite abs <=1e15."},
        Param{"integrality",Type::Array,false,Json::array(),{},"One continuous/integer/binary enum per variable; binary intersects bounds with [0,1].","",false,false,false,{},Type::String},
        Param{"maximize",Type::Boolean,false,false,{},"Maximize original c*x instead of minimizing."},num("time_limit",10.0,"Solver wall budget seconds, >0..30, bounded by process budget."),integer("node_limit",10000,"MILP node cap, 0..100000."),integer("maxiter",10000,"LP iteration cap, 1..100000."),num("mip_rel_gap",0.0,"MILP relative gap target, 0..0.1; actual mip_gap returned."),
        }};
    if(write)operation.params.emplace_back("out",Type::String,true,nullptr,std::vector<Json>{},"Exclusive .safetensors path; all nonempty output vectors become refs (outputs_in_artifact=true).");
    spec.operations.push_back(operation);
    }

    for(bool write:{false,true}){
    Operation operation{write?"ode_solve_to":"ode_solve",write?Access::Write:Access::Read,"Budgeted solve_ivp with closed RHS/event ASTs. Status OK/EVAL_LIMIT/TIME_LIMIT/STEP_LIMIT/EVENT_LIMIT/NON_FINITE/STEP_SIZE_TOO_SMALL/NO_CONVERGENCE. Non-OK trajectories are partial with termination failed. outputs_in_artifact identifies trajectory refs. Read form never writes; use ode_solve_to for large output.","object",{
        Param{"rhs",Type::Json,true,nullptr,{},"One scalar expression per state: list using y[i], t, params; or object matching named y0."},Param{"y0",Type::Json,true,nullptr,{},"Finite state array or named object; 1..32 states. Named states sorted by name."},
        Param{"states",Type::Array,false,nullptr,{},"Optional state aliases for array y0; unique, non-reserved, distinct from t/y/params.","",false,false,false,{},Type::String},Param{"params",Type::Object,false,Json::object(),{},"At most 16 named finite scalar parameters, default empty."},
        Param{"t_span",Type::Array,true,nullptr,{},"Two distinct finite times, absolute values <=1e12; forward or reverse.","",false,false,false,{},Type::Number},
        Param{"t_eval",Type::Json,false,nullptr,{},"Strictly ordered times within t_span; inline or npy/safetensors ref, <=10000. Missing endpoints are added and reported."},integer("max_output_points",101,"Uniform grid size when t_eval omitted, 2..10000. Duplicate representable times removed; effective_output_points reports the count. Output budget <=200000 numbers."),
        Param{"method",Type::String,false,"RK45",{"RK45","RK23","DOP853","Radau","BDF","LSODA"},"Closed solve_ivp method. Stiff problems: use BDF/Radau/LSODA; default RK45 can exhaust the 10s budget (TIME_LIMIT/STEP_LIMIT/EVAL_LIMIT, partial state)."},num("rtol",1e-6,"Relative tolerance, 1e-13..0.1."),Param{"atol",Type::Json,false,1e-9,{},"Absolute tolerance scalar or one per state, 1e-15..0.1."},num("max_step",nullptr,"Optional positive finite maximum step <=1e12."),
        Param{"events",Type::Array,false,Json::array(),{},"At most four {expr,direction:-1|0|1,terminal:boolean} events. Zeros at t_span[0] are not roots. event_index is integer; nonterminal cap 32 roots gives EVENT_LIMIT.","",false,false,false,{},Type::Object},
        integer("max_evals",200000,"Shared RHS-component/event expression cap, 1..200000. Counts finite-difference RHS calls too."),integer("max_steps",10000,"Accepted adaptive step cap, 1..20000."),num("time_limit",10.0,"Cooperative solve wall seconds, >0..30, checked on every expression evaluation."),
        }};
    if(write)operation.params.emplace_back("out",Type::String,true,nullptr,std::vector<Json>{},"Exclusive .safetensors path; time/states/events become refs (outputs_in_artifact=true).");
    spec.operations.push_back(operation);
    }

    spec.operations.push_back(Operation{"opt_minimize",Access::Read,"Minimise a scalar expression with scipy.optimize.minimize (closed method list). Statuses OK, NO_CONVERGENCE, EVAL_LIMIT, TIME_LIMIT, NON_FINITE; EVAL_LIMIT/TIME_LIMIT/NON_FINITE return the best point so far by objective value, which can be infeasible when constraints exist (see constraint_violation); gradients by finite differences unless gradient expressions are given (checked at x0, every evaluation counted). Result: par, fun, nit, nfev, evals, nonfinite_evals, constraint_violation","object",{
        Param{"objective",Type::String,true,nullptr,{},"Expression (<=2048 bytes, 512 nodes) in the unknowns and data columns; sum/mean/prod/min/max reducers only with data."},
        obj("x0",true,x0_text),
        Param{"method",Type::String,false,nullptr,{"Nelder-Mead","Powell","BFGS","L-BFGS-B","SLSQP","trust-constr"},"Default: SLSQP with constraints, L-BFGS-B with bounds, else BFGS. Bounds: not BFGS. Constraints: SLSQP, trust-constr. Gradient expressions: not Nelder-Mead/Powell."},
        obj("bounds",false,"{name: [lower, upper]} with null/\"-Inf\"/\"Inf\" for open ends; for a vector unknown one pair is applied to every component, or give per-component pairs [[lo0, hi0], [lo1, hi1], ...] (exactly one per component). lower < upper and x0 inside."),
        Param{"constraints",Type::Array,false,nullptr,{},"Up to 16 {\"type\":\"eq\"|\"ineq\",\"expr\":...}; ineq means expr >= 0.","",false,false,false,{},Type::Object},
        Param{"gradient",Type::Array,false,nullptr,{},"One expression per scalar unknown (sorted name order); checked against finite differences at x0.","",false,false,false,{},Type::String},
        obj("data",false,data_text),
        num("tol",nullptr,"Solver tolerance in (0, 1]; default is the method's own."),
        integer("maxiter",1000,"Iteration cap, 1..10000."),integer("max_evals",500000,"Cap on expression evaluations, 1..2000000; EVAL_LIMIT returns the best point so far."),
        Param{"nonfinite",Type::String,false,"inf",{"inf","error"},"inf: a non-finite objective counts as +Inf; error: stop with status NON_FINITE."}}});
    spec.operations.push_back(Operation{"opt_least_squares",Access::Read,"Nonlinear least squares with scipy.optimize.least_squares (trf/dogbox/lm): model y ~ f(data; params) or a residual expression list. Result: par, std_err (central-difference Jacobian with adaptive per-parameter steps: cov = (J'J)^-1 times 2*cost/(m-n) unless absolute_sigma; columns equilibrated and the Jacobian declared singular below a 1e-8 relative singular value, reported as cov_status SINGULAR_JACOBIAN with jacobian_condition), cost, rss, status OK/NO_CONVERGENCE/EVAL_LIMIT/TIME_LIMIT/NON_FINITE, cov_status","object",{
        Param{"model",Type::String,false,nullptr,{},"Expression in the unknowns and data columns; residual = (model - response)/sigma."},
        Param{"response",Type::String,false,nullptr,{},"Data column fitted by model."},
        Param{"residuals",Type::Array,false,nullptr,{},"Alternative to model: 1..64 residual expressions (each scalar or vector over data columns), concatenated.","",false,false,false,{},Type::String},
        obj("x0",true,x0_text),obj("data",false,data_text),obj("bounds",false,"As opt_minimize; not with method lm."),
        Param{"method",Type::String,false,"trf",{"trf","dogbox","lm"},"lm needs at least as many residuals as unknowns, no bounds, linear loss."},
        Param{"loss",Type::String,false,"linear",{"linear","soft_l1","huber","cauchy","arctan"},"Robust loss; std_err then uses the robust cost (documented)."},
        num("f_scale",1.0,"Loss scale, > 0."),
        Param{"sigma",Type::Json,false,nullptr,{},"model only: name of a positive data column or a positive number; residuals are divided by it."},
        Param{"absolute_sigma",Type::Boolean,false,false,{},"Treat sigma as absolute (no rescaling of the covariance by the residual variance)."},
        num("tol",1e-10,"ftol = xtol = gtol, in [2.3e-16, 0.1]."),integer("max_nfev",1000,"Residual evaluation budget for the solver, 1..100000."),integer("max_evals",500000,"Cap on expression evaluations, 1..2000000.")}});
    spec.operations.push_back(Operation{"num_quad",Access::Read,"One-dimensional QUADPACK integral (scipy.integrate.quad) of an expression in x over finite or infinite bounds. Statuses OK, NO_CONVERGENCE (message from QUADPACK), NON_FINITE, EVAL_LIMIT. Result: value, abserr, neval, subdivisions","object",{
        Param{"f",Type::String,true,nullptr,{},"Expression in x (no reducers, no data)."},
        Param{"a",Type::Json,true,nullptr,{},"Lower bound: finite number or \"-Inf\"/\"Inf\"."},Param{"b",Type::Json,true,nullptr,{},"Upper bound."},
        num("epsabs",1.49e-8,"Absolute tolerance in [0, 1]."),num("epsrel",1.49e-8,"Relative tolerance in [0, 1); at least 1e-13 when epsabs is 0."),
        integer("limit",50,"Maximum subdivisions, 1..1000."),
        Param{"points",Type::Array,false,nullptr,{},"Up to 64 break points strictly inside finite bounds.","",false,false,false,{},Type::Number},
        integer("max_evals",500000,"Cap on expression evaluations.")}});
    spec.operations.push_back(Operation{"opt_root",Access::Read,"Roots of n equations in n named unknowns (scipy.optimize.root hybr or lm), or a bracketed single equation (brentq): equations are expressions in the unknowns (the closed S3 grammar, no data columns, no eval path), each equation f_i(x) = 0. Honest success: the solver's own flag is not trusted, the residual max|f_i(x)| is recomputed and must be <= residual_tol, otherwise the status is NO_CONVERGENCE (a local minimum of the norm is not a root); EVAL_LIMIT and TIME_LIMIT return the best point. Statuses OK, NO_CONVERGENCE, EVAL_LIMIT, TIME_LIMIT, NON_FINITE. Result: x {name: value}, fun, residual_norm, residual_max, nfev, evals","object",{
        Param{"equations",Type::Array,true,nullptr,{},"1..16 expressions, one per unknown, each set equal to 0.","",false,false,false,{},Type::String},
        Param{"start",Type::Object,true,nullptr,{},"Start values {name: number} (names [a-z][a-z0-9_]{0,31}), one per equation; with bracket only the name is used.",},
        Param{"method",Type::String,false,nullptr,{"hybr","lm","brentq"},"hybr (Powell hybrid, square systems) or lm (Levenberg-Marquardt); default hybr without bracket, brentq with bracket; conflicting explicit method is BAD_PARAM."},
        Param{"bracket",Type::Array,false,nullptr,{},"One equation in one unknown: [lower, upper] with f(lower), f(upper) of opposite signs (BAD_INTERVAL otherwise); solved with brentq.","",false,false,false,{},Type::Number},
        Param{"xtol",Type::Number,false,1e-12,{},"Relative step tolerance in (0, 1) (hybr/lm) or absolute bracket width (brentq); default 1e-12. residual_tol is the binding final accuracy; loosening xtol alone cannot waive it."},
        Param{"residual_tol",Type::Number,false,1e-8,{},"Largest accepted max|f_i(x)| at the returned point; larger is NO_CONVERGENCE even when the solver claims success."},
        Param{"maxiter",Type::Integer,false,nullptr,{},"brentq-only iterations, 1..1000 (default 100); rejected with hybr/lm, which use max_evals."},Param{"max_evals",Type::Integer,false,500000,{},"Cap on expression evaluations, 1..2000000; EVAL_LIMIT returns the best point so far."}}});
    spec.operations.push_back(Operation{"num_root",Access::Read,"Bracketed scalar root of an expression in x (scipy.optimize brentq/bisect/ridder/toms748). f(lower) and f(upper) must be finite with opposite signs (BAD_INTERVAL otherwise). Statuses OK, NO_CONVERGENCE, EVAL_LIMIT. Result: root, f_root, iterations, function_calls","object",{
        Param{"f",Type::String,true,nullptr,{},"Expression in x (no reducers, no data)."},
        Param{"lower",Type::Number,true,nullptr,{},"Bracket start (finite)."},Param{"upper",Type::Number,true,nullptr,{},"Bracket end (finite, > lower)."},
        Param{"method",Type::String,false,"brentq",{"brentq","bisect","ridder","toms748"},"Bracketing method."},
        num("xtol",2e-12,"Absolute tolerance on the bracket width (not on |f|): a loose xtol returns a root with f_root far from 0 and status OK. Must be > 0."),num("rtol",8.881784197001252e-16,"Relative x tolerance, >= 4*eps."),integer("maxiter",100,"Iterations, 1..1000."),integer("max_evals",500000,"Cap on expression evaluations.")}});
    // S5: statsmodels. No formula strings: terms are a JSON list; the worker builds the design matrix and calls the array API.
    auto sm_source=[](){return std::vector<Param>{
        Param{"columns",Type::Object,false,nullptr,{},"Inline table {column: values}, <=1000 cells; strings, numbers, booleans, null for missing."},
        Param{"csv",Type::String,false,nullptr,{},"UTF-8 CSV file with a header, inside the working directory (<=16 MiB, <=200000 rows)."},
        Param{"frame",Type::String,false,nullptr,{},"Stored-only .npz table artifact from frame_import."}};};
    const char* terms_text="Design terms (no formula strings): [{\"const\":true|false}, {\"var\":col}, {\"cat\":col,\"ref\":level}, {\"interact\":[col,col(,col)]}, {\"poly\":[col,degree 2..6]}], at most 64. An intercept named const is included unless {\"const\":false}; its position follows the list. cat uses treatment coding with the explicit reference level and columns named col[T.level]; interact multiplies the factors' columns (name a:b); poly adds raw powers named col, col^2, ...";
    auto regression=[&](const char* name,bool write,bool glm){
        std::vector<Param> params=sm_source();
        params.push_back(Param{"y",Type::String,true,nullptr,{},"Response column."});
        params.push_back(Param{"terms",Type::Array,true,nullptr,{},terms_text,"",false,false,false,{},Type::Object});
        params.push_back(Param{"weights",Type::String,false,nullptr,{},glm?"Column of positive variance weights (var_weights).":"Column of positive weights: weighted least squares."});
        if(glm){
            params.push_back(Param{"family",Type::String,true,nullptr,{"gaussian","poisson","binomial","gamma"},"Exponential family. Links: gaussian identity|log|inverse_power; poisson log|identity|sqrt; binomial logit|probit|cloglog; gamma inverse_power|log|identity (first is the default)."});
            params.push_back(Param{"link",Type::String,false,nullptr,{"identity","log","inverse_power","logit","probit","cloglog","sqrt"},"Link function; must be one of the family's links."});
            params.push_back(Param{"offset",Type::String,false,nullptr,{},"Offset column added to the linear predictor."});
            params.push_back(Param{"exposure",Type::String,false,nullptr,{},"Exposure column (log link only): offset = log(exposure)."});
            params.push_back(Param{"maxiter",Type::Integer,false,100,{},"IRLS iterations, 1..100; NO_CONVERGENCE when exhausted."});
            params.push_back(Param{"tol",Type::Number,false,1e-8,{},"IRLS deviance tolerance, 1e-12..1e-2."});
        }
        params.push_back(Param{"cov_type",Type::String,false,"nonrobust",{"nonrobust","HC0","HC1","HC2","HC3","cluster"},"Covariance estimator. use_t defaults to true for nonrobust (OLS) and false otherwise, as in statsmodels."});
        params.push_back(Param{"cluster",Type::String,false,nullptr,{},"Cluster column for cov_type cluster."});
        params.push_back(Param{"use_t",Type::Boolean,false,nullptr,{},"Use t instead of normal reference distributions. Defaults: sm_ols true for nonrobust; sm_glm true for gaussian/gamma with nonrobust covariance (as R summary.glm), false otherwise."});
        params.push_back(Param{"alpha",Type::Number,false,0.05,{},"Confidence-interval level is 1-alpha."});
        params.push_back(Param{"na_action",Type::String,false,"error",{"error","drop"},"Missing (null/NaN) values in used columns: error (MISSING_VALUES) or drop the rows and report n_dropped. Infinities are always errors."});
        if(!write)params.push_back(Param{"include",Type::Array,false,nullptr,{},"fitted and/or residuals inline (counted against the 1000-number rule).","",false,false,false,{},Type::String});
        else params.push_back(Param{"out",Type::String,true,nullptr,{},"New .safetensors artifact with coef (p x 6: estimate, std_err, statistic, p_value, ci_lower, ci_upper), fitted and residuals (and row_index when rows were dropped)."});
        if(!write)params.back().enum_values={"fitted","residuals"};
        std::string title=glm?"Generalized linear model (statsmodels GLM, IRLS) on a validated JSON term list.":"Ordinary/weighted least squares (statsmodels OLS/WLS) on a validated JSON term list.";
        std::string statuses=glm?" Statuses OK, NO_CONVERGENCE, PERFECT_SEPARATION, RANK_DEFICIENT (standard errors NA; never masks the more severe statuses, see rank_deficient), TIME_LIMIT, NUMERICAL_FAILURE (non-finite output or a fitted mean outside the family support); coefficient table, deviance, null_deviance, pearson_chi2, scale, llf, aic, bic, converged, n_iter.":" Statuses OK, RANK_DEFICIENT (standard errors NA), TIME_LIMIT, NUMERICAL_FAILURE (overflowing sums of squares; standard errors NA); coefficient table, r2, r2_adj, f_statistic, llf, aic, bic, ssr, sigma, condition_number, durbin_watson. Tables wider than ~100 coefficients need the _to form.";
        return Operation{name,write?Access::Write:Access::Read,title+statuses+(write?" Writes coef/fitted/residuals to an exclusive safetensors artifact (CONFLICT if it exists).":""),"object",params};
    };
    spec.operations.push_back(regression("sm_ols",false,false));spec.operations.push_back(regression("sm_ols_to",true,false));
    spec.operations.push_back(regression("sm_glm",false,true));spec.operations.push_back(regression("sm_glm_to",true,true));
    auto arima=[&](const char* name,bool write){
        std::vector<Param> params=sm_source();
        params.push_back(Param{"series",Type::String,true,nullptr,{},"Time-ordered series column; missing values are modelled by the state-space filter (n_missing reported)."});
        params.push_back(Param{"order",Type::Array,true,nullptr,{},"[p, d, q]: p,q <= 5, d <= 2.","",false,false,false,{},Type::Integer});
        params.push_back(Param{"seasonal_order",Type::Array,false,nullptr,{},"[P, D, Q, s]: P,Q <= 2, D <= 1, s in 2..24; state dimension <= 64, <= 24 parameters.","",false,false,false,{},Type::Integer});
        params.push_back(Param{"trend",Type::String,false,"n",{"n","c","t","ct"},"Deterministic trend; c is the regression intercept (mean = c / (1 - sum phi) for AR models), not R's mean."});
        params.push_back(Param{"exog",Type::Array,false,nullptr,{},"Up to 8 regressor columns.","",false,false,false,{},Type::String});
        params.push_back(Param{"exog_future",Type::Object,false,nullptr,{},"{exog column: forecast_steps values} when forecasting with exog."});
        params.push_back(Param{"enforce_stationarity",Type::Boolean,false,true,{},"Constrain AR parameters to the stationary region."});
        params.push_back(Param{"enforce_invertibility",Type::Boolean,false,true,{},"Constrain MA parameters to the invertible region."});
        params.push_back(Param{"method",Type::String,false,"lbfgs",{"lbfgs","bfgs","nm","powell"},"Optimizer for the exact Gaussian likelihood."});
        params.push_back(Param{"maxiter",Type::Integer,false,200,{},"Optimizer iterations, 1..1000; NO_CONVERGENCE when the optimizer reports failure."});
        params.push_back(Param{"start_params",Type::Array,false,nullptr,{},"Optional starting parameters in statsmodels order (trend, exog, ar, ma, seasonal ar, seasonal ma, sigma2); the count is checked against the model.","",false,false,false,{},Type::Number});
        params.push_back(Param{"cov_type",Type::String,false,"opg",{"opg","oim","approx"},"Parameter covariance: opg (outer product of gradients, statsmodels default), oim (observed information) or approx (numerical Hessian of the likelihood, what R's arima reports)."});
        params.push_back(Param{"forecast_steps",Type::Integer,false,0,{},write?"Forecast horizon 0..1000 (artifact).":"Forecast horizon 0..100 (inline: mean, lower, upper)."});
        params.push_back(Param{"alpha",Type::Number,false,0.05,{},"Forecast interval and parameter CI level is 1-alpha."});
        if(write)params.push_back(Param{"out",Type::String,true,nullptr,{},"New .safetensors artifact with residuals and forecast (3 x h: mean, lower, upper)."});
        return Operation{name,write?Access::Write:Access::Read,std::string("SARIMAX exact maximum likelihood (statsmodels) on a numeric series. Statuses OK, NO_CONVERGENCE, TIME_LIMIT, NUMERICAL_FAILURE; params table (statsmodels names), sigma2, llf, aic, bic, hqic, nobs_effective, converged, forecast with normal intervals.")+(write?" Writes residuals and the forecast to an exclusive safetensors artifact.":""),"object",params};
    };
    spec.operations.push_back(arima("sm_arima",false));spec.operations.push_back(arima("sm_arima_to",true));
    {
        std::vector<Param> params=sm_source();
        params.push_back(Param{"series",Type::String,true,nullptr,{},"Series column (no missing values)."});
        params.push_back(Param{"kind",Type::String,true,nullptr,{"acf","pacf"},"Autocorrelation or partial autocorrelation."});
        params.push_back(Param{"nlags",Type::Integer,false,20,{},"1..100 and below the series length (pacf: below half of it)."});
        params.push_back(Param{"alpha",Type::Number,false,0.05,{},"Confidence level 1-alpha: Bartlett intervals for acf, +-z/sqrt(n) for pacf."});
        params.push_back(Param{"adjusted",Type::Boolean,false,false,{},"acf only: divide autocovariances by n-k instead of n (default matches R)."});
        params.push_back(Param{"method",Type::String,false,"ywm",{"ywm","ywadjusted","ols","ldb"},"pacf only: ywm (Yule-Walker with n denominator, R's pacf), ywadjusted, ols, ldb."});
        spec.operations.push_back(Operation{"sm_tsa_acf",Access::Read,"Sample autocorrelation or partial autocorrelation with confidence intervals (statsmodels.tsa.stattools); lags 0..nlags. Constant series are CONSTANT_SERIES.","object",params});
    }
    add_ml_ops(spec);
    {
        std::vector<Param> params=sm_source();
        params.push_back(Param{"series",Type::String,true,nullptr,{},"Series column (no missing values)."});
        params.push_back(Param{"test",Type::String,true,nullptr,{"acf","pacf","adfuller","kpss","ljung_box"},"acf / pacf: the sm_tsa_acf estimators (same code path); adfuller: augmented Dickey-Fuller unit-root test; kpss: stationarity test (null: stationary); ljung_box: Ljung-Box portmanteau test of no autocorrelation."});
        params.push_back(Param{"nlags",Type::Integer,false,nullptr,{},"acf / pacf only: 1..100 and below the series length (pacf: below half of it); default 20."});
        params.push_back(Param{"alpha",Type::Number,false,nullptr,{},"acf / pacf only: confidence level 1-alpha (default 0.05); rejected for the other tests."});
        params.push_back(Param{"adjusted",Type::Boolean,false,nullptr,{},"acf only: divide autocovariances by n-k instead of n (default false, matches R)."});
        params.push_back(Param{"method",Type::String,false,nullptr,{"ywm","ywadjusted","ols","ldb"},"pacf only: ywm (default, R's pacf), ywadjusted, ols, ldb."});
        params.push_back(Param{"regression",Type::String,false,nullptr,{"c","ct","ctt","n"},"adfuller: deterministic terms c (constant, default), ct (+ trend), ctt (+ quadratic trend) or n (none); kpss: c (level stationarity, default) or ct (trend stationarity)."});
        params.push_back(Param{"autolag",Type::String,false,nullptr,{"AIC","BIC","t-stat","none"},"adfuller: how the number of augmentation lags is chosen up to maxlag (default AIC); none uses exactly maxlag."});
        params.push_back(Param{"maxlag",Type::Integer,false,nullptr,{},"adfuller: largest number of augmentation lags, 0..1000 (default ceil(12 (n/100)^(1/4)), clipped to n//2 - deterministic_terms - 1); the series must be long enough (TOO_SHORT otherwise)."});
        params.push_back(Param{"kpss_lags",Type::Json,false,nullptr,{},"kpss: the Bartlett window length: \"auto\" (default, data dependent), \"legacy\" (ceil(12 (n/100)^(1/4))) or an integer 0..1000."});
        params.push_back(Param{"lags",Type::Json,false,nullptr,{},"ljung_box: an integer m (tests lags model_df+1..m) or a strictly increasing unique list of up to 100 lags in 1..100; default 10."});
        params.push_back(Param{"model_df",Type::Integer,false,nullptr,{},"ljung_box: degrees of freedom removed from the chi-square reference (fitted ARMA parameters), integer lags m: 0 <= model_df < m; explicit list: below every lag; default 0."});
        spec.operations.push_back(Operation{"sm_tsa_tools",Access::Read,"Time-series tests and correlograms on one numeric series (statsmodels.tsa.stattools / stats.diagnostic; patsy is never imported): acf and pacf (delegating to sm_tsa_acf), adfuller (statistic, MacKinnon p-value, used_lag, nobs, critical_values, icbest), kpss (statistic, interpolated p-value with p_value_bounded none|above|below, above/below when the statistic is outside the published table, lags, critical_values) and ljung_box (lags, statistic, p_value per lag). Statuses OK, NUMERICAL_FAILURE, TIME_LIMIT; constant series are CONSTANT_SERIES, series too short for the requested lags TOO_SHORT.","object",params});
    }
    for(auto& operation:spec.operations){bool guarded=operation.name.rfind("sm_",0)==0||operation.name.rfind("ml_",0)==0||s7_op(operation.name)||operation.name.rfind("opt_",0)==0||operation.name.rfind("num_",0)==0||operation.name.rfind("ode_",0)==0||operation.name=="frame_mutate"||operation.name=="frame_merge"||operation.name=="frame_rolling"||s8b_op(operation.name)||graph_op(operation.name)||operation.name=="array_to_frame"||operation.name=="frame_to_array_to"||operation.name=="frame_to_array"||operation.name=="stats_dist"||operation.name=="stats_dist_to";if(guarded&&std::none_of(operation.params.begin(),operation.params.end(),[](const Param& p){return p.name=="time_limit";}))operation.params.emplace_back("time_limit",Type::Number,false,10.,std::vector<Json>{},"Cooperative compute wall budget >0..30 seconds (default 10); independent hard child cap PYCLI_TIMEOUT_S <=50 seconds includes imports. Native calls also have pre-spawn work/output caps.");}
    for(auto& op:spec.operations){if(op.access==Access::Write)op.description+=" Successful file writes return outputs_in_artifact=true and artifacts[name] descriptors with ref, path, format, key, dtype, shape, nbytes (key/dtype/shape null for whole files).";std::string name=op.name;op.handler=[name](const Params& p){return call(name,p);};for(auto& param:op.params)if(param.description.empty())param.description=param.name+" input; see README for shapes and limits.";}
    return run_cli(spec,argc,argv);
}
