// Bounded, stored-only NPZ preflight. No decompressor or caller dtype dispatch.
#pragma once
namespace frames {
constexpr uint64_t ROWS=200000, BYTES=256*1024*1024, COLS=256, CELLS=500000, CSV_BYTES=16*1024*1024, STRING_BYTES=65536;
inline void bad(const std::string& s){invalid(s,"BAD_DATA");}
inline uint64_t le(const std::string& b,size_t p,size_t n){if(p>b.size()||n>b.size()-p)bad("truncated table ZIP");uint64_t v=0;for(size_t i=0;i<n;++i)v|=uint64_t(static_cast<unsigned char>(b[p+i]))<<(8*i);return v;}
inline std::string bounded_read(const std::string& path){std::ifstream f(path,std::ios::binary);std::string b;char buf[65536];while(f){f.read(buf,sizeof buf);auto n=f.gcount();if(b.size()+n>BYTES)bad("table file exceeds 256 MiB");b.append(buf,n);}if(!f.eof())bad("cannot read table");return b;}
inline void utf8(const std::string& s){
 for(size_t i=0;i<s.size();){unsigned char c=s[i++];if(c<128)continue;size_t n;uint32_t code,minimum;
 if(c>=0xc2&&c<=0xdf){n=1;code=c&31;minimum=128;}else if(c>=0xe0&&c<=0xef){n=2;code=c&15;minimum=2048;}else if(c>=0xf0&&c<=0xf4){n=3;code=c&7;minimum=65536;}else bad("invalid UTF-8");
 if(n>s.size()-i)bad("truncated UTF-8");for(size_t j=0;j<n;++j){unsigned char d=s[i++];if((d&192)!=128)bad("invalid UTF-8 continuation");code=(code<<6)|(d&63);}if(code<minimum||code>0x10ffff||(code>=0xd800&&code<=0xdfff))bad("invalid UTF-8 codepoint");
 }
}
inline void names(const Json& n){if(!n.is_array()||n.empty()||n.size()>COLS)bad("need 1..256 columns");std::set<std::string> seen;for(auto& v:n){if(!v.is_string())bad("column name must be string");std::string s=v;if(s.empty()||s.size()>256||s.find('\0')!=std::string::npos||!seen.insert(s).second)bad("empty, duplicate or oversized column name");}}
inline Json npz(const std::string& b){
 Json metadata;
 if(b.size()<22||b.size()>BYTES)bad("invalid table ZIP size");size_t end=b.size()-22;
 if(le(b,end,4)!=0x06054b50||le(b,end+4,2)||le(b,end+6,2)||le(b,end+20,2))bad("table requires single-disk ZIP without comments");
 auto count=le(b,end+10,2),size=le(b,end+12,4),offset=le(b,end+16,4);
 if(count!=le(b,end+8,2)||!count||count>2*COLS+1||offset>end||size!=end-offset)bad("invalid table ZIP directory");
 size_t p=offset;std::set<std::string> seen;uint64_t total=0;std::vector<std::pair<uint64_t,uint64_t>> spans;
 for(size_t i=0;i<count;++i){
  if(le(b,p,4)!=0x02014b50||le(b,p+8,2)||le(b,p+10,2))bad("table ZIP must be unencrypted and stored-only (no compressed members)");
  auto compressed=le(b,p+20,4),bytes=le(b,p+24,4),nl=le(b,p+28,2),extra=le(b,p+30,2),comment=le(b,p+32,2),local=le(b,p+42,4);
  if(bytes!=compressed||bytes>BYTES||total>BYTES-bytes||!nl||nl>64||extra||comment||le(b,p+34,2))bad("table ZIP member cap/format");total+=bytes;
  if(p+46+nl> end)bad("truncated ZIP name");std::string name=b.substr(p+46,nl);
  if(name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.")!=std::string::npos)bad("invalid table member name bytes");
  if(name.find('/')!=std::string::npos||name.find('\\')!=std::string::npos||!seen.insert(name).second||name.size()<5||name.substr(name.size()-4)!=".npy")bad("invalid table member name");
  if(le(b,local,4)!=0x04034b50||le(b,local+6,2)||le(b,local+8,2)||le(b,local+26,2)!=nl)bad("invalid table local header");
  auto lextra=le(b,local+28,2);if(lextra!=0&&lextra!=20)bad("unexpected local ZIP extra");
  if(local+30+nl>offset||b.substr(local+30,nl)!=name)bad("ZIP local name mismatch");
  if(lextra){auto e=local+30+nl;if(le(b,e,2)!=1||le(b,e+2,2)!=16||le(b,e+4,8)!=bytes||le(b,e+12,8)!=bytes)bad("invalid local ZIP64 sizes");}
  else if(le(b,local+18,4)!=bytes||le(b,local+22,4)!=bytes)bad("ZIP member size mismatch");
  auto data=local+30+nl+lextra;if(data>offset||bytes>offset-data||bytes<10)bad("table member out of bounds");spans.emplace_back(local,data+bytes);
  if(b.compare(data,6,"\x93NUMPY")||le(b,data+6,1)!=1||le(b,data+7,1))bad("table NPY requires v1");
  auto h=le(b,data+8,2);if(!h||h>65536||10+h>bytes)bad("invalid table NPY header");auto parsed=parse_npy_header(b.substr(data+10,h));
  if(parsed.shape.size()!=1)bad("table arrays must have rank one");uint64_t n=parsed.shape[0],width=0;
  if(parsed.descr=="<i8"||parsed.descr=="<f8")width=8;
  else if(parsed.descr=="|b1"||parsed.descr=="|u1")width=1;
  else if(parsed.descr.rfind("|S",0)==0){auto s=parsed.descr.substr(2);if(s.empty()||s.size()>7||s.find_first_not_of("0123456789")!=std::string::npos)bad("invalid UTF-8 byte array dtype");width=std::stoull(s);}
  else bad("object or unsupported table array dtype");
  if(!width||width>STRING_BYTES||(name!="__pycli_table__.npy"&&n>ROWS)||n>BYTES/width||10+h+n*width!=bytes)bad("table array cap/payload mismatch");
  if(name=="__pycli_table__.npy"){
   if(parsed.descr!="|u1"||n>1024*1024)bad("invalid metadata array");
   try{metadata=Json::parse(b.substr(data+10+h,n),[](int depth,Json::parse_event_t,Json&){if(depth>32)bad("table metadata nesting cap");return true;});}catch(const Json::exception&){bad("invalid table metadata JSON");}
   if(!metadata.is_object()||(!metadata.contains("version")||metadata["version"]!=1)||!metadata.contains("columns")||!metadata.contains("n_rows")||!metadata["n_rows"].is_number_unsigned()||metadata["n_rows"].get<uint64_t>()>ROWS)bad("invalid table metadata rows/version");names(metadata["columns"]);
  }
  p+=46+nl;
 }
 if(p!=end||!seen.count("__pycli_table__.npy"))bad("missing table metadata or trailing directory bytes");
 std::sort(spans.begin(),spans.end());uint64_t next=0;for(auto& s:spans){if(s.first!=next)bad("ZIP member overlap or gap");next=s.second;}if(next!=offset)bad("unexpected bytes before directory");
 std::set<std::string> expected={"__pycli_table__.npy"};for(size_t i=0;i<metadata["columns"].size();++i){expected.insert("c"+std::to_string(i)+".npy");expected.insert("m"+std::to_string(i)+".npy");}if(seen!=expected)bad("unexpected table members");
 if(metadata["n_rows"].get<uint64_t>()*metadata["columns"].size()>CELLS)bad("table exceeds 500000 cells");
 return metadata;
}
inline void csv(const std::string& b,const std::string& delimiter,uint64_t maxrows){
 if(delimiter.size()!=1||delimiter[0]=='"'||delimiter[0]=='\r'||delimiter[0]=='\n'||delimiter[0]=='\0')bad("delimiter must be one non-control byte");
 if(b.size()>CSV_BYTES)invalid("CSV exceeds 16MiB byte budget","INPUT_TOO_LARGE");
 utf8(b);
 bool quoted=false,after=false,start=true;uint64_t field=0,fields=1,rows=0,expected=0;std::string value;Json header=Json::array();
 auto finishfield=[&]{if(rows==0)header.push_back(value);value.clear();field=0;start=true;after=false;};
 auto finishrow=[&]{if(rows&&start&&fields==1&&field<=1){field=0;return;}finishfield();if(!rows){expected=fields;names(header);}else if(fields!=expected)bad("ragged CSV");if(rows++>maxrows)invalid("CSV row cap exceeded","INPUT_TOO_LARGE");if((rows-1)*expected>CELLS)invalid("CSV exceeds 500000 cell budget","INPUT_TOO_LARGE");fields=1;};
 size_t i=b.compare(0,3,"\xef\xbb\xbf")==0?3:0;bool pending=false;
 for(;i<b.size();++i){char c=b[i];if(c=='\0')bad("NUL in CSV");pending=true;if(++field>STRING_BYTES+2)invalid("CSV string width exceeds 65536 bytes","INPUT_TOO_LARGE");
  if(quoted){if(c=='"'){if(i+1<b.size()&&b[i+1]=='"'){if(!rows)value+='"';++i;}else{quoted=false;after=true;}}else if(!rows)value+=c;continue;}
  if(c==delimiter[0]){finishfield();if(++fields>COLS)bad("CSV column cap exceeded");continue;}
  if(c=='\r'||c=='\n'){if(c=='\r'){if(i+1>=b.size()||b[i+1]!='\n')bad("CSV requires LF or CRLF");++i;}finishrow();pending=false;continue;}
  if(after)bad("text after quoted CSV cell");if(c=='"'){if(!start)bad("quote in unquoted CSV field");quoted=true;start=false;continue;}start=false;if(!rows)value+=c;
 }
 if(quoted)bad("unterminated CSV quote");if(pending)finishrow();if(!rows)bad("CSV needs header");
}
inline void predicate(const Json& node,size_t depth=0,size_t* budget=nullptr){size_t own=0;if(!budget)budget=&own;if(depth>32||++*budget>256||!node.is_object())bad("predicate depth/node cap");
 for(auto key:{"and","or","not"})if(node.contains(key)){if(node.size()!=1)bad("invalid predicate keys");if(std::string(key)=="not"){predicate(node[key],depth+1,budget);return;}auto& a=node[key];if(!a.is_array()||a.empty())bad("empty predicate junction");for(auto& n:a)predicate(n,depth+1,budget);return;}
 if(!node.contains("col")||!node["col"].is_string()||!node.contains("op")||!node["op"].is_string())bad("invalid predicate leaf");std::string op=node["op"];
 std::set<std::string> ops={"eq","ne","lt","le","gt","ge","isin","isnull","between","startswith","contains"};if(!ops.count(op)||node.size()!=(op=="isnull"?2:3)||(op!="isnull"&&!node.contains("value")))bad("unknown predicate operator/keys");
 if(op=="isnull")return;auto& v=node["value"];if(op=="isin"||op=="between"){if(!v.is_array()||v.size()>1000||(op=="between"&&v.size()!=2))bad("invalid predicate values");for(auto& x:v)if(x.is_array()||x.is_object())bad("predicate requires scalar values");}
 else if(v.is_array()||v.is_object()||((op=="contains"||op=="startswith")&&!v.is_string()))bad("invalid predicate value");
}
inline void rule(const std::string& r){size_t i=0;while(i<r.size()&&r[i]>='0'&&r[i]<='9')++i;if(!i||i>5||r[0]=='0'||std::stoul(r.substr(0,i))>10000)bad("invalid resample rule count");std::set<std::string> units={"min","h","D","W","MS","ME","QS","YS"};if(!units.count(r.substr(i)))bad("invalid resample unit");}
inline void install(const std::string& from,const std::string& to){auto checked=confined(to,true);if(checked!=to)bad("output parent changed");auto b=bounded_read(from);int fd=open(fs::path(to).parent_path().c_str(),O_TMPFILE|O_WRONLY,0600);if(fd<0)throw std::runtime_error("cannot create anonymous table output");try{size_t pos=0;while(pos<b.size()){ssize_t n=write(fd,b.data()+pos,b.size()-pos);if(n<0&&errno==EINTR)continue;if(n<=0)throw std::runtime_error("cannot write output");pos+=n;}mode_t mask=umask(0);umask(mask);if(fchmod(fd,0644&~mask)||fsync(fd))throw std::runtime_error("cannot flush output");auto source="/proc/self/fd/"+std::to_string(fd);if(linkat(AT_FDCWD,source.c_str(),AT_FDCWD,to.c_str(),AT_SYMLINK_FOLLOW)){if(errno==EEXIST)throw Error{ErrorCode::Conflict,"output already exists",{{"path",to}}};throw std::runtime_error("cannot install output");}int dir=open(fs::path(to).parent_path().c_str(),O_RDONLY|O_DIRECTORY);if(dir<0||fsync(dir)){if(dir>=0)close(dir);unlink(to.c_str());throw std::runtime_error("cannot sync output directory");}close(dir);close(fd);}catch(...){close(fd);throw;}}
}
