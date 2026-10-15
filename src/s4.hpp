// S4 closed validators. Caller expressions are parsed here; no caller source crosses the child boundary.
#pragma once
constexpr size_t LP_VARIABLES=512, LP_ROWS=4096, LP_ENTRIES=200000, ODE_STATES=32, ODE_POINTS=10000, S4_OUTPUT_NUMBERS=200000;
double s4_number(const Json& p,const char* key,double low,double high){double x=p.at(key).get<double>();if(!std::isfinite(x)||x<low||x>high)bad_param(std::string(key)+" out of range");return x;}
Json prepare_linprog(const Params& in,std::map<std::string,Array>& arrays){
 Json p=in;arrays["c"]=load(in.at("c"));const auto& c=arrays["c"];if(c.shape.size()!=1||c.data.size()>LP_VARIABLES)invalid("LP needs 1..512 variables","INPUT_TOO_LARGE");need_finite(c,"c");size_t n=c.data.size(),entries=0,rows=0;
 for(auto suffix:{"ub","eq"}){std::string a="a_"+std::string(suffix),b="b_"+std::string(suffix);if(in.contains(a)!=in.contains(b))bad_param(a+" and "+b+" must be supplied together");if(!in.contains(a))continue;
  arrays[a]=load(in.at(a));arrays[b]=load(in.at(b));const auto& aa=arrays[a];const auto& bb=arrays[b];need_finite(aa,a.c_str());need_finite(bb,b.c_str());
  if(aa.shape.size()!=2||bb.shape.size()!=1||size_t(aa.shape[1])!=n||size_t(aa.shape[0])!=bb.data.size())bad_param("LP matrix/vector dimensions disagree");entries+=aa.data.size();rows+=bb.data.size();
 }
 if(entries>LP_ENTRIES||rows>LP_ROWS)invalid("LP allows <=200000 combined matrix entries and <=4096 rows","INPUT_TOO_LARGE");
 Json bounds=Json::array();auto read_end=[](const Json& x,bool upper){if(x.is_null()||(x.is_string()&&x==(upper?"Inf":"-Inf")))return upper?std::numeric_limits<double>::infinity():-std::numeric_limits<double>::infinity();if(!x.is_number()||!std::isfinite(x.get<double>()))bad_param("LP bounds must be finite, null or signed infinity tokens");return x.get<double>();};
 if(in.contains("bounds")&&(!in["bounds"].is_array()||in["bounds"].size()!=n))bad_param("bounds need one [lower,upper] per variable");
 Json kinds=in.value("integrality",Json::array());if(!kinds.is_array()||(!kinds.empty()&&kinds.size()!=n))bad_param("integrality needs one continuous/integer/binary enum per variable");
 Json integers=Json::array();
 for(size_t i=0;i<n;++i){double lo=0,hi=INFINITY;if(in.contains("bounds")){const auto& pair=in["bounds"][i];if(!pair.is_array()||pair.size()!=2)bad_param("bounds need pairs");lo=read_end(pair[0],false);hi=read_end(pair[1],true);}
  std::string kind=kinds.empty()?"continuous":kinds[i].is_string()?kinds[i].get<std::string>():"invalid";if(kind!="continuous"&&kind!="integer"&&kind!="binary")bad_param("integrality enum is continuous/integer/binary");if(kind=="binary"){lo=std::max(lo,0.0);hi=std::min(hi,1.0);}if(lo>hi)bad_param("inconsistent bounds");
  bounds.push_back({std::isinf(lo)?Json(nullptr):Json(lo),std::isinf(hi)?Json(nullptr):Json(hi)});integers.push_back(kind=="continuous"?0:1);
 }
 p["bounds"]=bounds;p["integrality"]=integers;s4_number(in,"time_limit",1e-9,30);s4_number(in,"mip_rel_gap",0,0.1);int_param(in,"node_limit",0,100000);int_param(in,"maxiter",1,100000);
 if(in.contains("out"))p["out"]=output_path(in);return p;
}
Json prepare_ode(const Params& in,std::map<std::string,Array>& arrays){
 Json p=in;std::vector<double> initial;std::vector<std::string> labels;std::set<std::string> names={"t","y"};
 if(in["y0"].is_object()){
  for(auto it=in["y0"].begin();it!=in["y0"].end();++it)labels.push_back(it.key());std::sort(labels.begin(),labels.end());
  for(const auto& name:labels){if(!in["y0"][name].is_number())bad_param("initial states must be finite numbers");initial.push_back(in["y0"][name].get<double>());}
  if(in.contains("states"))bad_param("states is only used with array y0");
 }else if(in["y0"].is_array()){
  for(const auto& x:in["y0"]){if(!x.is_number())bad_param("initial states must be finite numbers");initial.push_back(x.get<double>());}
  if(in.contains("states"))labels=in["states"].get<std::vector<std::string>>();
 }else bad_param("y0 needs an array or named object");
 size_t n=initial.size();if(!n||n>ODE_STATES)invalid("ODE needs 1..32 states","INPUT_TOO_LARGE");for(double x:initial)if(!std::isfinite(x))bad_param("initial states must be finite");
 if(!labels.empty()&&labels.size()!=n)bad_param("states must match y0 length");for(const auto& name:labels){check_variable_name(name,"state");if(!names.insert(name).second)bad_param("duplicate/reserved state name");}
 const auto& params=in.at("params");if(!params.is_object()||params.size()>16)bad_param("params needs <=16 named finite numbers");for(auto it=params.begin();it!=params.end();++it){check_variable_name(it.key(),"parameter");if(!names.insert(it.key()).second)bad_param("parameter name collides with t/y/state");if(!it.value().is_number()||!std::isfinite(it.value().get<double>()))bad_param("params values must be finite numbers");}
 Json asts=Json::array();const auto& rhs=in["rhs"];std::map<std::string,size_t> dims={{"y",n}};
 if(rhs.is_array()){if(rhs.size()!=n)bad_param("one RHS expression required per state");for(const auto& x:rhs){if(!x.is_string())bad_param("RHS must be expressions");asts.push_back(parse_expression(x,names,dims,false,"rhs"));}}
 else if(rhs.is_object()&&!labels.empty()){if(rhs.size()!=n)bad_param("RHS names must match state names");for(const auto& name:labels){if(!rhs.contains(name)||!rhs[name].is_string())bad_param("RHS names must match state names");asts.push_back(parse_expression(rhs[name],names,dims,false,"rhs."+name));}}
 else bad_param("RHS must be a list or named state expression object");
 Json events=Json::array();if(in["events"].size()>4)bad_param("at most four events");for(const auto& e:in["events"]){if(!e.is_object()||!e.contains("expr")||!e["expr"].is_string()||e.size()>1+size_t(e.contains("direction"))+size_t(e.contains("terminal")))bad_param("event requires {expr,direction?,terminal?}");
  int direction=0;if(e.contains("direction")){if(!e["direction"].is_number_integer())bad_param("event direction must be -1/0/1");if(e["direction"]!= -1&&e["direction"]!=0&&e["direction"]!=1)bad_param("event direction must be -1/0/1");direction=e["direction"].get<int>();if(direction< -1||direction>1)bad_param("event direction must be -1/0/1");}if(e.contains("terminal")&&!e["terminal"].is_boolean())bad_param("event terminal must be boolean");events.push_back({{"ast",parse_expression(e["expr"],names,dims,false,"events")},{"direction",direction},{"terminal",e.value("terminal",false)}});
 }
 const auto& span=in["t_span"];if(!span.is_array()||span.size()!=2)bad_param("t_span needs two finite times");double t[2];for(int k=0;k<2;++k){if(!span[k].is_number()||!std::isfinite(span[k].get<double>())||std::abs(span[k].get<double>())>1e12)bad_param("t_span times must be finite with abs <=1e12");t[k]=span[k].get<double>();}if(t[0]==t[1])bad_param("t_span endpoints must differ");
 std::vector<double> grid;Json added=Json::array();if(in.contains("t_eval")){auto a=load(in["t_eval"]);need_finite(a,"t_eval");if(a.shape.size()!=1||a.data.size()>ODE_POINTS)invalid("t_eval needs <=10000 points","INPUT_TOO_LARGE");grid=a.data;double sign=t[1]>t[0]?1:-1;
  for(size_t i=0;i<grid.size();++i)if(sign*(grid[i]-t[0])<0||sign*(t[1]-grid[i])<0||(i&&sign*(grid[i]-grid[i-1])<=0))bad_param("t_eval must be strictly ordered within t_span");
  if(grid.front()!=t[0]){grid.insert(grid.begin(),t[0]);added.push_back(t[0]);}if(grid.back()!=t[1]){grid.push_back(t[1]);added.push_back(t[1]);}
 }else{size_t count=int_param(in,"max_output_points",2,ODE_POINTS);for(size_t i=0;i<count;++i)grid.push_back(t[0]+(t[1]-t[0])*(double(i)/double(count-1)));grid.back()=t[1];}
 if(grid.size()>ODE_POINTS||(grid.size()+1)*(n+1)+128*(n+2)+n> S4_OUTPUT_NUMBERS)invalid("ODE output number budget exceeded","INPUT_TOO_LARGE");
 int_param(in,"max_evals",1,200000);int_param(in,"max_steps",1,20000);s4_number(in,"time_limit",1e-9,30);s4_number(in,"rtol",1e-13,0.1);
 if(in["atol"].is_array()){if(in["atol"].size()!=n)bad_param("atol vector must match state count");for(const auto& x:in["atol"])if(!x.is_number()||!std::isfinite(x.get<double>())||x.get<double>()<1e-15||x.get<double>()>0.1)bad_param("atol out of range");}else s4_number(in,"atol",1e-15,0.1);
 if(in.contains("max_step"))s4_number(in,"max_step",std::numeric_limits<double>::denorm_min(),1e12);
 if(in.contains("out"))p["out"]=output_path(in);p.erase("rhs");p.erase("y0");p.erase("t_eval");p["rhs_asts"]=asts;p["events"]=events;p["initial"]=initial;p["state_names"]=labels;p["grid"]=grid;p["endpoints_added"]=added;return p;
}
