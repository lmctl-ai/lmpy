// Exercise the actual CLI parsers under ASan/UBSan with structure-aware ZIP mutations.
#define main pycli_entry_for_fuzz
#include "../src/main.cpp"
#undef main
#include <random>
#include <iostream>
void put(std::string& b,size_t p,uint64_t v,size_t n){for(size_t i=0;i<n;++i)b[p+i]=char(v>>(8*i));}
std::string seed(){
 std::string body="{\"version\":1,\"n_rows\":0,\"columns\":[\"x\"]}";std::string b,central;size_t count=0;
 for(auto name:{"c0.npy","m0.npy","__pycli_table__.npy"}){
  bool meta=std::string(name)=="__pycli_table__.npy";std::string n=name,h="{'descr': '|u1', 'fortran_order': False, 'shape': ("+std::to_string(meta?body.size():0)+",), }\n";
  std::string a="\x93NUMPY\x01";a.push_back(0);a.resize(10);put(a,8,h.size(),2);a+=h;if(meta)a+=body;
  size_t local=b.size();std::string l(30,0);put(l,0,0x04034b50,4);put(l,18,a.size(),4);put(l,22,a.size(),4);put(l,26,n.size(),2);b+=l+n+a;
  std::string c(46,0);put(c,0,0x02014b50,4);put(c,20,a.size(),4);put(c,24,a.size(),4);put(c,28,n.size(),2);put(c,42,local,4);central+=c+n;++count;
 }
 size_t offset=b.size();b+=central;size_t e=b.size();b.resize(e+22);put(b,e,0x06054b50,4);put(b,e+8,count,2);put(b,e+10,count,2);put(b,e+12,central.size(),4);put(b,e+16,offset,4);return b;
}
int main(){std::mt19937_64 rng(2026);auto good=seed();frames::npz(good);size_t accepted=0,rejected=0;for(int i=0;i<10000;++i){auto b=good;size_t c=good.find("PK\x01\x02");switch(i%5){case 0:break;case 1:{const int fields[]={8,10,20,24,28,30,32,34,42};put(b,c+fields[rng()%9],rng(),2);break;}case 2:b[rng()%b.size()]=char(rng());break;case 3:b.resize(rng()%b.size());break;case 4:put(b,b.size()-6,rng(),4);break;}try{frames::npz(b);++accepted;}catch(const lmcpp::Error&){++rejected;}}
if(accepted<2000||rejected<1000)return 1;
for(int i=0;i<10000;++i){try{frames::csv(std::string(rng()%100,char(rng())),",",100);}catch(const lmcpp::Error&){}try{frames::rule(std::to_string(rng())+"D");}catch(const lmcpp::Error&){}Json node={{"col","x"},{"op",i%2?"eq":"contains"},{"value",std::string(rng()%100,char(rng()))}};frames::predicate(node);}
size_t s4_ok=0,s4_bad=0;
for(int i=0;i<2000;++i){
 Json lp={{"c",{1.0,2.0}},{"bounds",{{0,1},{0,1}}},{"integrality",Json::array()},{"time_limit",10.0},{"node_limit",10000},{"maxiter",10000},{"mip_rel_gap",0.0},{"maximize",false}};
 Json ode={{"rhs",{"y[0]+t"}},{"y0",{1.0}},{"params",Json::object()},{"events",Json::array()},{"t_span",{0,1}},{"method","RK45"},{"max_output_points",101},{"max_evals",200000},{"max_steps",10000},{"time_limit",10.0},{"rtol",1e-6},{"atol",1e-9}};
 switch(i%6){case 0:break;case 1:lp["node_limit"]=uint64_t(rng());ode["t_span"]={0,double(rng())};break;case 2:lp["bounds"][0]={2,1};ode["rhs"]={"y[999999]"};break;case 3:lp["integrality"]={"binary","integer"};ode["events"]={{{"expr","t-0.5"},{"direction",-1}}};break;case 4:lp["a_ub"]={{1}};lp["b_ub"]={1};ode["max_output_points"]=int(rng()%20000);break;case 5:lp["integrality"]={"semi","continuous"};ode["params"]={{"y",1}};break;}
 for(bool is_ode:{false,true}){std::map<std::string,Array> arrays;try{if(is_ode)prepare_ode(ode,arrays);else prepare_linprog(lp,arrays);++s4_ok;}catch(const lmcpp::Error&){++s4_bad;}}
}
if(s4_ok<1000||s4_bad<1000)return 1;
size_t ml_ok=0,ml_bad=0;
for(int i=0;i<2000;++i){
 pycli::TensorBundle model;Json prep={{"scale",false},{"one_hot",Json::array()},{"categories",Json::object()}};
 model.metadata={{"pycli_model","1"},{"family","linear"},{"sklearn_version","1.7.2"},{"features","[\"x\"]"},{"feature_dtypes","{\"x\":\"numeric-f64\"}"},{"preprocess",prep.dump()},{"params","{\"fit_intercept\":true}"},{"labels","[]"},{"width","1"},{"train_rows","3"},{"seed","null"},{"components","1"},{"target","y"},{"train_sha256",std::string(64,'a')},{"lock_id",std::string(64,'b')}};
 pycli::TensorData t;t.dtype="F64";t.shape={1};t.bytes.resize(8);model.tensors["coef_"]=t;model.tensors["intercept_"]=t;
 switch(i%6){case 0:break;case 1:model.tensors["coef_"].shape={int64_t(rng())};break;case 2:model.metadata["preprocess"]=std::string(rng()%100,char(rng()));break;case 3:model.metadata["width"]=std::to_string(rng());break;case 4:model.tensors["unexpected"]=t;break;case 5:model.metadata["features"]="null";break;}
 try{ml_model_info(model,false);++ml_ok;}catch(const lmcpp::Error&){++ml_bad;}
 Json decoded;try{frames::csv(std::string(rng()%100,char(rng())),",",20000,&decoded);}catch(const lmcpp::Error&){}
 try{ml_table_npz(good);}catch(const lmcpp::Error&){}
}
// Structural class integrity plus the short canonical-label parser under sanitizers.
for(int i=0;i<1000;++i){
 pycli::TensorBundle model;Json prep={{"scale",false},{"one_hot",Json::array()},{"categories",Json::object()}};
 model.metadata={{"pycli_model","1"},{"family","knn_classifier"},{"sklearn_version","1.7.2"},{"features","[\"x\"]"},{"feature_dtypes","{\"x\":\"numeric-f64\"}"},{"preprocess",prep.dump()},{"params","{\"n_neighbors\":1,\"weights\":\"uniform\"}"},{"labels","[\"a\",\"b\",\"c\"]"},{"width","1"},{"train_rows","3"},{"seed","null"},{"components","1"},{"target","y"},{"train_sha256",std::string(64,'a')},{"lock_id",std::string(64,'b')}};
 pycli::TensorData x;x.dtype="F64";x.shape={3,1};x.bytes.resize(24);model.tensors["train_x_"]=x;
 pycli::TensorData y;y.dtype="I64";y.shape={3};y.bytes.resize(24);for(int64_t j=0;j<3;++j)std::memcpy(y.bytes.data()+j*8,&j,8);model.tensors["classes_"]=y;
 bool mutate=i%2;if(mutate)y.bytes[8]^=1;model.tensors["train_y_"]=y;
 bool rejected=false;try{ml_model_info(model,false);}catch(const lmcpp::Error&){rejected=true;}if(rejected!=mutate)return 1;
 ml_integer_label(std::string(rng()%100,char(rng())));
}
if(ml_ok<300||ml_bad<1000)return 1;
std::cout<<"S6 model/CSV preflight fuzz: "<<ml_ok<<" accepted, "<<ml_bad<<" rejected\n";
std::cout<<"S4 structured preflight fuzz: "<<s4_ok<<" accepted, "<<s4_bad<<" rejected\n";
std::cout<<"table fuzz: "<<accepted<<" ZIPs passed all size/header/layout checks; "<<rejected<<" rejected; 10000 CSV/rule/predicate cases\n";
}
