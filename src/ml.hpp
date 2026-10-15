// S6 closed ML protocol. All caller-dependent sizes and model headers checked before spawn.
#pragma once
#include "ml_spec.hpp"
const Json& ml_spec(){static const Json s=Json::parse(PYCLI_ML_SPEC);return s;}
bool ml_integer_label(const std::string& text){
 if(text=="0")return true;if(text.empty()||text.size()>11)return false;size_t start=text[0]=='-'?1:0;
 if(start==text.size()||text[start]<'1'||text[start]>'9'||text.find_first_not_of("0123456789",start)!=std::string::npos)return false;
 return std::abs(std::stoll(text))<=1000000000LL;
}
void ml_name(const Json& v){if(!v.is_string())bad_param("ML column names must be strings");std::string s=v;if(s.empty()||s.size()>64||s.find('\0')!=std::string::npos)bad_param("ML names must be 1..64 UTF-8 bytes");frames::utf8(s);}
Json ml_params(const std::string& family,const Json& supplied){
 const auto& families=ml_spec()["families"];if(!families.contains(family))bad_param("unknown ML family");
 const auto& table=families[family]["params"];if(!supplied.is_object())bad_param("params must be an object");Json out=Json::object();
 for(auto i=supplied.begin();i!=supplied.end();++i)if(!table.contains(i.key()))bad_param("unknown parameter for "+family+": "+i.key());
 for(auto i=table.begin();i!=table.end();++i){auto rule=i.value();auto v=supplied.value(i.key(),rule["default"]);std::string type=rule["type"];
  if(type=="boolean"){if(!v.is_boolean())bad_param(i.key()+" must be boolean");}
  else if(type=="string"){if(!v.is_string()||std::find(rule["enum"].begin(),rule["enum"].end(),v)==rule["enum"].end())bad_param("invalid "+i.key());}
  else{if(!v.is_number()||(type=="integer"&&!v.is_number_integer())||!std::isfinite(v.get<double>())||v<rule["min"]||v>rule["max"])bad_param("out-of-range "+i.key());}
  out[i.key()]=v;
 }return out;
}
// Decode only after the stored-only NPZ parser has checked every span, dtype and size.
Json ml_table_npz(const std::string& b){try{
 Json meta=frames::npz(b);uint64_t rows=meta["n_rows"];if(rows<1||rows>20000||rows*meta["columns"].size()>200000)invalid("ML table cap: 2..20000 rows /200000 cells","INPUT_TOO_LARGE");
 struct Member{std::string type;size_t start,width,count;};std::map<std::string,Member> members;size_t p=0;
 while(frames::le(b,p,4)==0x04034b50){size_t nl=frames::le(b,p+26,2),extra=frames::le(b,p+28,2),n=frames::le(b,p+18,4);size_t d=p+30+nl+extra;if(extra)n=frames::le(b,p+30+nl+4,8);size_t h=frames::le(b,d+8,2);auto a=parse_npy_header(b.substr(d+10,h));size_t width=a.descr=="<i8"||a.descr=="<f8"?8:a.descr.rfind("|S",0)==0?std::stoull(a.descr.substr(2)):1;members[b.substr(p+30,nl)]={a.descr,d+10+h,width,size_t(a.shape[0])};p=d+n;}
 Json cols=Json::object();for(size_t c=0;c<meta["columns"].size();++c){std::string key=meta["columns"][c];auto a=members.at("c"+std::to_string(c)+".npy"),mask=members.at("m"+std::to_string(c)+".npy");if(a.count!=rows||mask.count!=rows||mask.type!="|b1")bad_param("BAD_MODEL table row/mask mismatch");Json values=Json::array();auto policy=meta["column_meta"].at(key);std::string dtype=policy["dtype"];
  if(dtype!="int64"&&dtype!="float64"&&dtype!="bool"&&dtype!="string"&&dtype!="category")bad_param("ML rejects datetime features; encode explicitly");
  for(size_t r=0;r<rows;++r){if((unsigned char)b.at(mask.start+r)){values.push_back(nullptr);continue;}size_t d=a.start+r*a.width;if(a.type=="<f8"){double v;std::memcpy(&v,b.data()+d,8);if(!std::isfinite(v))values.push_back(nullptr);else values.push_back(v);}else if(a.type=="<i8"){int64_t v;std::memcpy(&v,b.data()+d,8);if(dtype=="category"){if(v<0||v>=int64_t(policy["categories"].size()))bad_param("invalid category code");values.push_back(policy["categories"][v]);}else values.push_back(v);}else if(a.type=="|b1")values.push_back(bool(b.at(d)));else{std::string s=b.substr(d,a.width);s.resize(s.find('\0')==std::string::npos?s.size():s.find('\0'));frames::utf8(s);values.push_back(s);}}
  cols[key]=values;
 }return cols;
 }catch(const Error&){throw;}catch(const std::exception&){bad_param("invalid ML table metadata/payload");}
 return Json();
}
Json ml_source(const Params& p){
 if(p.count("columns")+p.count("records")+p.count("csv")+p.count("frame")!=1)bad_param("provide exactly one of columns/records/csv/frame");
 if(p.contains("columns"))return p["columns"];
 if(p.contains("records")){Json cols=Json::object();auto records=p["records"];if(!records.is_array()||records.empty())bad_param("records required");if(!records[0].is_object())bad_param("record must be object");for(auto i=records[0].begin();i!=records[0].end();++i)cols[i.key()]=Json::array();for(auto& r:records){if(!r.is_object()||r.size()!=cols.size())bad_param("ragged records");for(auto i=cols.begin();i!=cols.end();++i){if(!r.contains(i.key()))bad_param("ragged records");i.value().push_back(r[i.key()]);}}return cols;}
 if(p.contains("frame"))return ml_table_npz(frames::bounded_read(confined(p["frame"]),32*1024*1024));
 auto bytes=frames::bounded_read(confined(p["csv"]),frames::CSV_BYTES);Json decoded;frames::csv(bytes,",",20000,&decoded);return decoded;
}
size_t ml_columns(const Json& cols,const Json& features,const Json& prep,const std::string& target,bool classification,const std::string& groups=""){
 if(!cols.is_object()||cols.empty()||cols.size()>256)bad_param("columns must be a nonempty map");size_t rows=cols.begin().value().size();if(!rows||rows>20000||rows*cols.size()>200000)invalid("ML cap: 20000 rows /200000 cells","INPUT_TOO_LARGE");
 for(auto i=cols.begin();i!=cols.end();++i)if(!i.value().is_array()||i.value().size()!=rows)bad_param("ragged columns");
 if(!features.is_array()||features.empty()||features.size()>64)bad_param("features must contain 1..64 names");
 if(!prep.is_object()||prep.size()!=2||!prep.contains("scale")||!prep["scale"].is_boolean()||!prep.contains("one_hot")||!prep["one_hot"].is_array())bad_param("preprocess requires {scale:boolean,one_hot:[names]}");
 std::set<std::string> hot,seen;for(auto& v:prep["one_hot"]){ml_name(v);if(!hot.insert(v.get<std::string>()).second)bad_param("duplicate one_hot feature");}
 size_t width=0;auto scalar=[&](const Json& v,bool text){if(v.is_null())bad_param("ML does not accept missing/NaN/Inf values");if(text){if(v.is_string()){if(v.get<std::string>().size()>64)bad_param("ML labels/categories capped at 64 UTF-8 bytes");frames::utf8(v);}else if(!v.is_number_integer()||std::abs(v.get<double>())>1e9)bad_param("classification labels/categories must be strings or integers with abs<=1e9");}else if(v.is_boolean()){}else if(!v.is_number()||!std::isfinite(v.get<double>())||std::abs(v.get<double>())>1e12)bad_param("ML numeric values must be finite, abs<=1e12");};
 for(auto& v:features){ml_name(v);std::string name=v;if(!seen.insert(name).second||!cols.contains(name))bad_param("duplicate or missing feature: "+name);std::set<std::string> levels;for(auto& cell:cols[name]){scalar(cell,hot.count(name));if(hot.count(name))levels.insert(cell.dump());}width+=hot.count(name)?levels.size():1;}
 for(auto& h:hot)if(!seen.count(h))bad_param("one_hot must name a feature");if(width>64||rows*width>200000)invalid("encoded features exceed 64 /200000 cells","INPUT_TOO_LARGE");
 for(auto& name:{target,groups})if(!name.empty()){ml_name(name);if(!cols.contains(name))bad_param("missing target/group: "+name);std::set<std::string> levels;for(auto& v:cols[name]){scalar(v,name==groups||classification);levels.insert(v.dump());}if(name==target&&classification&&(levels.size()<2||levels.size()>32))bad_param("classification needs 2..32 classes");}
 return width;
}
[[noreturn]] void bad_model(const std::string& message){invalid(message,"BAD_MODEL");}
Json ml_model_info(const pycli::TensorBundle& b,bool skew){
 try{
  static const std::set<std::string> keys={"pycli_model","family","sklearn_version","features","feature_dtypes","preprocess","params","labels","width","train_rows","components","seed","target","train_sha256","lock_id"};std::set<std::string> got;for(auto& v:b.metadata)got.insert(v.first);if(keys!=got)bad_model("BAD_MODEL metadata keys");
  if(b.metadata.at("pycli_model")!="1"||(!skew&&b.metadata.at("sklearn_version")!="1.7.2"))invalid("model/sklearn version mismatch; explicit allow_version_skew required","MODEL_VERSION_MISMATCH");Json m;for(auto& v:b.metadata)m[v.first]=(v.first=="features"||v.first=="feature_dtypes"||v.first=="preprocess"||v.first=="params"||v.first=="labels"||v.first=="width"||v.first=="train_rows"||v.first=="seed"||v.first=="components")?Json::parse(v.second):Json(v.second);
  for(const char* key:{"train_sha256","lock_id"}){std::string h=m[key];if(h.size()!=64||h.find_first_not_of("0123456789abcdef")!=std::string::npos)bad_model("BAD_MODEL provenance digest");}
  std::string f=m["family"];if(!ml_spec()["families"].contains(f)||ml_spec()["families"][f]["attributes"].empty())bad_model("MODEL_NOT_PERSISTABLE");if(ml_params(f,m["params"])!=m["params"])bad_model("BAD_MODEL params");
  auto dim=[&](const char* key,int cap){auto v=m.at(key);if(!v.is_number_integer()||v.get<int64_t>()<1||v.get<int64_t>()>cap)bad_model("BAD_MODEL dimension cap");return v.get<int>();};int w=dim("width",64),n=dim("train_rows",20000),k=m["labels"].size(),r=dim("components",64);if(w<1||w>64||n<2||n>20000||w*n>200000||r<1||r>64||m["features"].size()>64||m["features"].empty()||m["labels"].size()>32||m.dump().size()>16384)bad_model("BAD_MODEL dimensions/metadata cap");
  if(!m["features"].is_array()||!m["labels"].is_array())bad_model("BAD_MODEL features/labels");if(!m["seed"].is_null()&&(!m["seed"].is_number_integer()||m["seed"].get<int64_t>()<0||m["seed"].get<int64_t>()>4294967295LL))bad_model("BAD_MODEL seed");std::set<std::string> labelset;for(auto& l:m["labels"]){if((!l.is_string()&&!l.is_number_integer())||(l.is_string()&&(l.get<std::string>().size()>64||l.get<std::string>().find('\0')!=std::string::npos))||!labelset.insert(l.dump()).second)bad_model("BAD_MODEL class labels");}
  std::string task=ml_spec()["families"][f]["task"];if((task=="classification"&&k<2)||(task!="classification"&&k!=0))bad_model("BAD_MODEL class count");
  std::set<std::string> names;for(auto& x:m["features"]){ml_name(x);if(!names.insert(x.get<std::string>()).second)bad_model("BAD_MODEL feature names");}
  auto prep=m["preprocess"];if(!prep.is_object()||prep.size()!=3||!prep.contains("scale")||!prep["scale"].is_boolean()||!prep.contains("one_hot")||!prep["one_hot"].is_array()||!prep.contains("categories")||!prep["categories"].is_object())bad_model("BAD_MODEL preprocessing");
  Json dummy=Json::object();for(auto& name:m["features"]){std::string s=name;dummy[s]=Json::array({1,2});}Json base={{"scale",prep["scale"]},{"one_hot",prep["one_hot"]}};int width=m["features"].size();std::set<std::string> hs;for(auto& x:prep["one_hot"]){if(!x.is_string()||!names.count(x.get<std::string>())||!hs.insert(x.get<std::string>()).second||!prep["categories"].contains(x.get<std::string>()))bad_model("BAD_MODEL one-hot names");auto cat=prep["categories"][x.get<std::string>()];if(!cat.is_array()||cat.empty()||cat.size()>64)bad_model("BAD_MODEL categories");std::set<std::string> unique;for(auto& l:cat){if(!l.is_string()||(l.get<std::string>().size()>64||l.get<std::string>().find('\0')!=std::string::npos)||!unique.insert(l.get<std::string>()).second)bad_model("BAD_MODEL category levels");}width+=cat.size()-1;}
  if(!m["feature_dtypes"].is_object()||m["feature_dtypes"].size()!=names.size())bad_model("BAD_MODEL feature dtype map");for(auto& name:names)if(m["feature_dtypes"].value(name,std::string())!=(hs.count(name)?"string":"numeric-f64"))bad_model("BAD_MODEL feature dtype policy");
  if(prep["categories"].size()!=hs.size()||width!=w)bad_model("BAD_MODEL encoded width");
  std::map<std::string,std::vector<int64_t>> shapes;auto attrs=ml_spec()["families"][f]["attributes"];for(auto i=attrs.begin();i!=attrs.end();++i){std::string t=i.value();std::vector<int64_t> shape;if(t=="p")shape={w};else if(t=="one")shape={1};else if(t=="classes")shape={k};else if(t=="cp")shape={k,w};else if(t=="decision")shape={k==2?1:k,w};else if(t=="decision_rows")shape={k==2?1:k};else if(t=="np")shape={n,w};else if(t=="n")shape={n};else if(t=="kp")shape={m["params"]["n_clusters"].get<int>(),w};else if(t=="rp")shape={r,w};else if(t=="r"||t=="lda_components")shape={r};else if(t=="scalings")shape={w,r};else bad_model("BAD_MODEL attribute schema");shapes[i.key()]=shape;}
  if(prep["scale"]==true){shapes["pp_mean_"]={w};shapes["pp_scale_"]={w};}if(b.tensors.size()!=shapes.size())bad_model("BAD_MODEL unexpected/missing attribute");
  for(auto& v:b.tensors){if(!shapes.count(v.first)||v.second.shape!=shapes[v.first])bad_model("BAD_MODEL wrong attribute shape");size_t count=1;for(auto d:v.second.shape)count*=d;if(v.second.bytes.size()!=count*8)bad_model("BAD_MODEL tensor bytes");bool integer=v.first=="classes_"||(f=="knn_classifier"&&v.first=="train_y_");if(v.second.dtype!=(integer?"I64":"F64"))bad_model("BAD_MODEL wrong attribute dtype");for(size_t j=0;j<v.second.bytes.size()/8;++j){if(integer){int64_t x;std::memcpy(&x,v.second.bytes.data()+j*8,8);if(x<0||x>=k||(v.first=="classes_"&&x!=int64_t(j)))bad_model("BAD_MODEL class codes");}else{double x;std::memcpy(&x,v.second.bytes.data()+j*8,8);if(!std::isfinite(x)||(v.first=="train_x_"&&std::abs(x)>1e12))bad_model("BAD_MODEL nonfinite/out-of-range attribute");if((v.first=="pp_scale_"||v.first=="scale_"||(v.first=="var_"&&f=="gaussian_nb"))&&x<=0)bad_model("BAD_MODEL nonpositive scale/variance");}}}
  if(f=="knn_classifier"){std::set<int64_t> codes;const auto& bytes=b.tensors.at("train_y_").bytes;for(size_t j=0;j<bytes.size();j+=8){int64_t code;std::memcpy(&code,bytes.data()+j,8);codes.insert(code);}if(codes.size()!=size_t(k))bad_model("BAD_MODEL training labels disagree with stored classes");}
  auto params=m["params"];if((params.contains("n_neighbors")&&params["n_neighbors"].get<int>()>n)||(params.contains("n_clusters")&&params["n_clusters"].get<int>()>n)||(f=="lda"&&params["n_components"].get<int>()>std::min(w,k-1)))bad_model("BAD_MODEL parameters exceed training dimensions");
  if((f=="pca"&&(r!=m["params"]["n_components"].get<int>()||r>std::min(w,n)))||(f=="lda"&&(k<2||r>std::min(w,k-1))))bad_model("BAD_MODEL component count");return m;
 }catch(const Error&){throw;}catch(const std::exception&){bad_model("BAD_MODEL invalid metadata or attributes");}
 return Json();
}
Json prepare_ml(const std::string& op,const Params& in,std::map<std::string,Array>& arrays,std::string& predicted){
 Json p=in;Json cols=ml_source(in);bool predict=op=="ml_predict";std::string family;
 if(predict){pycli::TensorBundle b;try{auto path=confined(in["model"]);if(fs::file_size(path)>32*1024*1024)bad_param("BAD_MODEL exceeds 32MiB");b=pycli::read_bundle(path);}catch(const Error&){bad_model("invalid/truncated safetensors");}catch(const std::exception&){bad_model("invalid/truncated safetensors");}Json m=ml_model_info(b,in.value("allow_version_skew",false));family=m["family"];p["family"]=family;p["features"]=m["features"];p["preprocess"]={{"scale",m["preprocess"]["scale"]},{"one_hot",m["preprocess"]["one_hot"]}};p["model_info"]=m;p.erase("model");Json attrs=Json::object();int count=0;for(auto& t:b.tensors){Array a;a.shape=t.second.shape;size_t n=t.second.bytes.size()/8;a.data.resize(n);for(size_t i=0;i<n;++i){if(t.second.dtype=="I64"){int64_t v;std::memcpy(&v,t.second.bytes.data()+i*8,8);a.data[i]=v;}else std::memcpy(&a.data[i],t.second.bytes.data()+i*8,8);}std::string key="m_"+std::to_string(count++);arrays[key]=std::move(a);attrs[t.first]=key+"_file";}p["model_attrs"]=attrs;
 }else{family=p["family"];p["params"]=ml_params(family,p.value("params",Json::object()));auto spec=ml_spec()["families"][family];if(spec["stochastic"]==true&&!p.contains("seed"))bad_param("seed is required for stochastic ML families");if(op=="ml_cv"&&!p.contains("seed"))bad_param("seed is required for CV");}
 if(p.contains("seed")&&(!p["seed"].is_number_integer()||p["seed"].get<int64_t>()<0||p["seed"].get<int64_t>()>4294967295LL))bad_param("seed must be uint32");
 std::string task=ml_spec()["families"][family]["task"],target=p.value("target",std::string()),groups=p.value("groups",std::string());bool supervised=task=="classification"||task=="regression";
 if(!predict&&supervised&&target.empty())bad_param("target required");if(!predict&&!supervised&&!target.empty()&&op!="ml_cv")bad_param("target is not used by this family");
 if(in.contains("csv")){
  Json types=Json::object();std::set<std::string> strings;for(auto& h:p["preprocess"].value("one_hot",Json::array()))if(h.is_string())strings.insert(h.get<std::string>());if(task=="classification"&&!target.empty())strings.insert(target);if(!groups.empty())strings.insert(groups);
  std::set<std::string> nums;for(auto& f:p["features"])if(f.is_string()&&!strings.count(f.get<std::string>()))nums.insert(f.get<std::string>());if(task=="regression"&&!target.empty())nums.insert(target);
  for(auto& name:nums)if(cols.contains(name)){types[name]="float64";for(auto& v:cols[name]){if(!v.is_string())bad_param("CSV numeric field required");double number=csv_number(v.get<std::string>());if(!std::isfinite(number))bad_param("ML rejects NaN/Inf");v=number;}}
  for(auto& name:strings)types[name]="string";p["dtypes"]=types;
 }
 // Canonical decimal class labels have the same type in CSV, JSON and table sources.
 if(!predict&&task=="classification"&&cols.contains(target)&&cols[target].is_array()&&!cols[target].empty()){
  bool strings=cols[target][0].is_string();for(auto& v:cols[target])if(v.is_string()!=strings)bad_param("class labels must have one type");
  if(strings){bool integers=true;for(auto& v:cols[target]){auto text=v.get<std::string>();if(!ml_integer_label(text)){integers=false;break;}}
   if(integers){for(auto& v:cols[target])v=std::stoll(v.get<std::string>());if(in.contains("csv"))p["dtypes"][target]="int64";}
  }
 }
 size_t width=ml_columns(cols,p["features"],p["preprocess"],predict?"":target,task=="classification",groups),rows=cols.begin().value().size();if(!predict&&rows<2)bad_param("at least two training rows required");
 // File numeric strings are parsed strictly here and typed explicitly below by ml_source's CSV policy.
 size_t folds=op=="ml_cv"?p["n_splits"].get<size_t>():1;if(folds<1||folds>20||folds>rows)bad_param("n_splits must be 2..20 and <=rows");
 if(predict){width=p["model_info"]["width"].get<size_t>();if(rows*width>200000)invalid("prediction encoded cells exceed 200000","INPUT_TOO_LARGE");size_t outwidth=1;std::string kind=p["kind"];size_t classes=p["model_info"]["labels"].size();if(kind=="predict_proba")outwidth=classes;else if(kind=="decision_function")outwidth=classes==2?1:classes;else if(kind=="transform")outwidth=family=="standard_scaler"?width:family=="kmeans"?p["model_info"]["params"]["n_clusters"].get<size_t>():p["model_info"]["components"].get<size_t>();if(rows*outwidth>200000)invalid("prediction output exceeds 200000 cells","INPUT_TOO_LARGE");}
 auto params=predict?p["model_info"]["params"]:p["params"];uint64_t mult=params.value("max_iter",1)*uint64_t(params.value("n_init",1))*uint64_t(params.value("n_clusters",1))*uint64_t(params.value("n_estimators",1))*uint64_t(params.value("max_depth",1));
 if(family=="pca"||family=="lda"||family=="linear"||family=="ridge")mult=width;
 if(!predict&&task=="classification"&&(family=="logistic"||family=="linear_svc"||family.rfind("hist_gradient",0)==0||family.rfind("random_forest",0)==0)){std::set<std::string> classes;for(auto& y:cols[target])classes.insert(y.dump());mult*=classes.size();}
 if(predict)mult=std::max<size_t>(1,p["model_info"]["labels"].size());
 if(uint64_t(rows)*width*mult*folds>200000000)invalid("ML work budget exceeds 200M (rows*encoded_features*iterations*folds)","INPUT_TOO_LARGE");
 if(family.rfind("knn_",0)==0&&op=="ml_cv"&&rows*rows>5000000)invalid("KNN CV pair budget exceeds 5M","INPUT_TOO_LARGE");
 if(!predict){if(params.contains("n_components")&&params["n_components"].get<int>()>int(std::min(rows,width)))bad_param("n_components exceeds rows/features");if(params.contains("n_clusters")&&params["n_clusters"].get<int>()>int(rows))bad_param("n_clusters exceeds rows");if(params.contains("n_neighbors")&&params["n_neighbors"].get<int>()>int(rows))bad_param("n_neighbors exceeds rows");}
 bool tree=ml_spec()["families"][family]["attributes"].empty();
 if(op=="ml_fit"){if(tree){if(p.contains("out")||!p.contains("predict_frame")||!p.contains("predict_out"))invalid("trees require predict_frame/predict_out and no model out","MODEL_NOT_PERSISTABLE");predicted=frames::bounded_read(confined(p["predict_frame"]),32*1024*1024);auto pc=ml_table_npz(predicted);ml_columns(pc,p["features"],p["preprocess"],"",false);if(pc.begin().value().size()*width>200000)invalid("tree prediction encoded cell cap","INPUT_TOO_LARGE");p["out"]=confined(p["predict_out"],true);if(fs::path(p["out"].get<std::string>()).extension()!=".npz")bad_param("predict_out requires .npz");p.erase("predict_frame");p.erase("predict_out");}
 else{if(!p.contains("out"))bad_param("persistable ml_fit needs out");p["out"]=output_path(p);if(p.contains("predict_frame")||p.contains("predict_out"))bad_param("predict_frame is only for nonpersistable tree families; use ml_predict");}}
 if(predict){std::string kind=p["kind"];bool cls=task=="classification";if((kind=="predict"&&(family=="pca"||family=="standard_scaler"))||(kind=="predict_proba"&&(!cls||family=="linear_svc"))||(kind=="decision_function"&&family!="logistic"&&family!="linear_svc"&&family!="lda")||(kind=="transform"&&family!="pca"&&family!="standard_scaler"&&family!="kmeans"&&family!="lda"))bad_param("prediction kind unavailable for family");p["out"]=confined(p["out"],true);if(fs::path(p["out"].get<std::string>()).extension()!=".npz")bad_param("ml_predict out requires .npz");if(family.rfind("knn_",0)==0&&rows*p["model_info"]["train_rows"].get<size_t>()>5000000)invalid("KNN prediction pair budget exceeds 5M","INPUT_TOO_LARGE");}
 if(op=="ml_cv"){if(folds<2)bad_param("CV needs 2..20 splits");if(p["cv"]=="group"&&groups.empty())bad_param("group CV requires groups column");if(p["cv"]!="group"&&!groups.empty())bad_param("groups only used for group CV");if(p["cv"]=="timeseries"&&p["shuffle"]==true)bad_param("timeseries cannot shuffle");if(p["cv"]=="stratified"&&task!="classification")bad_param("stratified requires classifier");if(!supervised)bad_param("S6 CV supports supervised families only");std::set<std::string> allowed=task=="classification"?std::set<std::string>{"accuracy","f1_macro","roc_auc","log_loss"}:std::set<std::string>{"mse","mae","r2"};if(!p["metrics"].is_array()||p["metrics"].empty()||p["metrics"].size()>4)bad_param("metrics needs 1..4 entries");std::set<std::string> seen;for(auto& m:p["metrics"])if(!m.is_string()||!allowed.count(m.get<std::string>())||!seen.insert(m.get<std::string>()).second)bad_param("invalid/duplicate metric");}
 double budget=p.value("time_limit",10.);if(!(budget>0&&budget<=30))bad_param("time_limit must be >0..30");p["ml_spec"]=ml_spec();return p;
}
std::string ml_parameter_description(){
 std::string out="Closed per-family primitive parameter tables; unknown keys rejected before spawn. ";
 for(auto f=ml_spec()["families"].begin();f!=ml_spec()["families"].end();++f){out+=f.key()+": ";if(f.value()["params"].empty())out+="no parameters";for(auto p=f.value()["params"].begin();p!=f.value()["params"].end();++p){auto v=p.value();out+=p.key()+" ("+v["type"].get<std::string>()+", default "+v["default"].dump();if(v.contains("min"))out+=", range ["+v["min"].dump()+","+v["max"].dump()+"]";if(v.contains("enum"))out+=", enum "+v["enum"].dump();out+="); ";}out+=". ";}return out;
}
void add_ml_ops(CliSpec& spec){
 std::vector<Json> families;for(auto i=ml_spec()["families"].begin();i!=ml_spec()["families"].end();++i)families.push_back(i.key());
 auto source=[](){return std::vector<Param>{Param{"columns",Type::Object,false},Param{"records",Type::Array,false,nullptr,{},"Inline records <=1000 cells","",false,false,false,{},Type::Object},Param{"csv",Type::String,false},Param{"frame",Type::String,false}};};
 for(const std::string op:{"ml_fit","ml_predict","ml_cv"}){auto p=source();if(op=="ml_predict"){p.emplace_back("model",Type::String,true);p.emplace_back("kind",Type::String,false,"predict",std::vector<Json>{"predict","predict_proba","decision_function","transform"});p.emplace_back("allow_version_skew",Type::Boolean,false,false);p.emplace_back("out",Type::String,true);}else{p.emplace_back("family",Type::String,true,nullptr,families);p.emplace_back("features",Type::Array,true,nullptr,std::vector<Json>{},"Ordered feature names","",false,false,false,std::vector<std::string>{},Type::String);p.emplace_back("target",Type::String,false);p.emplace_back("params",Type::Object,false,Json::object());p.emplace_back("preprocess",Type::Object,true);p.emplace_back("seed",Type::Integer,false);if(op=="ml_fit"){p.emplace_back("out",Type::String,false);p.emplace_back("predict_frame",Type::String,false);p.emplace_back("predict_out",Type::String,false);}else{p.emplace_back("cv",Type::String,false,"kfold",std::vector<Json>{"kfold","stratified","group","timeseries"});p.emplace_back("n_splits",Type::Integer,false,5);p.emplace_back("shuffle",Type::Boolean,false,false);p.emplace_back("groups",Type::String,false);p.emplace_back("metrics",Type::Array,true,nullptr,std::vector<Json>{},"Closed metrics by task","",false,false,false,std::vector<std::string>{},Type::String);}}
 p.emplace_back("time_limit",Type::Number,false,10.);
 const std::map<std::string,std::string> descriptions={
 {"columns","Exactly one source: inline column-name -> equal-length scalar arrays, <=1000 cells; no missing values. Canonical integer class strings become integers; leading zeros/+ remain identifiers."},
 {"records","Exactly one source: inline record objects, <=1000 cells; identical columns per row; no missing values."},
 {"csv","Exactly one source: confined UTF-8 header CSV path, <=16MiB /20000 rows /200000 cells. Numeric features typed float64, categorical features string; canonical decimal class labels typed int64."},
 {"frame","Exactly one source: confined S2 table NPZ path <=32MiB /20000 rows /200000 cells; no datetime or missing values."},
 {"model","Confined v1 safetensors model path <=32MiB; strict attribute shapes/types, class consistency, feature order and sklearn version checked before spawn."},
 {"kind","predict: labels/values; predict_proba: columns in echoed class order; decision_function: binary positive for classes[1]; transform: PCA/scaler/kmeans/LDA. Family availability checked."},
 {"allow_version_skew","Default false: require exact sklearn version; true explicitly accepts version skew without a compatibility guarantee."},
 {"family","Closed estimator family enum. Trees fit-and-predict only (no persisted model); all n_jobs=1, no caller solvers/callables."},
 {"features","Required ordered unique feature names, 1..64 raw/encoded; model preserves this order independently of column-map order."},
 {"target","Required column name for classification/regression; forbidden for unsupervised fit. Class labels homogeneous string/integer, 2..32 classes; canonical decimal strings normalize to int64."},
 {"params",ml_parameter_description()},
 {"preprocess","Required exact object {scale:boolean, one_hot:[unique feature names]}; no defaults or extra keys. Sorted full one-hot encoding then optional training mean/population-variance scaling; <=64 encoded features; unseen levels rejected."},
 {"seed","uint32 [0,4294967295], required for stochastic families (linear_svc/kmeans/trees) and every CV; optional for deterministic lbfgs logistic and other families. MT19937 echoed only when randomness is used."},
 {"out","Exclusive atomic output path: ml_fit persistable model requires .safetensors; ml_predict requires table .npz. Readonly forbids writes; existing files conflict."},
 {"predict_frame","Required only for nonpersistable tree fit: S2 table NPZ with matching features, <=200000 encoded cells; predictions performed in the same call."},
 {"predict_out","Required exclusive .npz prediction artifact for tree fit; mutually exclusive with model out."},
 {"cv","kfold/stratified/group/timeseries; supervised families only; preprocessing learned independently per fold. Stratified requires classification; timeseries cannot shuffle."},
 {"n_splits","Integer 2..20, <= rows; default 5; group CV also needs at least n_splits distinct groups."},
 {"shuffle","Default false; shuffle kfold/stratified/group with required seed; forbidden for timeseries."},
 {"groups","Required group column for group CV; forbidden for other CV kinds; group identities never overlap train/test."},
 {"metrics","Required unique metric names (1..4): regressors mse/mae/r2; classifiers accuracy/f1_macro/binary roc_auc/log_loss. Probability metrics unavailable for linear_svc; undefined folds return non-OK with diagnostics."},
 {"time_limit","ML computation wall budget >0..30 seconds, default 10; imports excluded; Python signal boundaries plus hard child timeout and pre-spawn work caps."}};
 for(auto& param:p)param.description=descriptions.at(param.name);
 spec.operations.push_back(Operation{op,op=="ml_cv"?Access::Read:Access::Write,"Bounded scikit-learn: explicit preprocessing, strict v1 safetensors, single-thread sequential execution; no missing values. See docs/s6.md for family parameters and caps.","object",p});}
}
