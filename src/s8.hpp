// S8a front end: table/array bridge, frame_mutate / frame_merge / frame_rolling, ml_metrics and seeded sampling (stats_dist rvs).
// Every table is preflighted here (bounded stored-only NPZ, metadata, dtypes, names, caps) and every expression is parsed here, so the worker only ever
// sees a validated request and a copy of the table in a private temp directory.
#pragma once
constexpr size_t S8_INLINE=1000, S8_VALUES=200000, S8_COLS=256, S8_CELLS=500000, S8_RVS_INLINE=1000, S8_RVS_FILE=1000000, S8_WORK=100000000, S8_CLASSES=30, S8_ASSIGN=32, S8_WINDOW=100000;
constexpr double S8_EXACT=9007199254740992.0;  // 2^53
struct S8Extra{std::map<std::string,std::string> tables;std::string table_out;bool tensor_write=false,needs_spec=false;};
bool s8_op(const std::string& op,const Params& in){
    static const std::set<std::string> fixed={"frame_to_array","frame_to_array_to","array_to_frame","frame_mutate","frame_merge","frame_rolling","ml_metrics","stats_dist_to"};
    return fixed.count(op)||(op=="stats_dist"&&in.contains("fn")&&in["fn"]=="rvs");
}
struct S8Table{Json meta;std::string bytes;};
S8Table s8_table(const Params& in,const char* key){
    auto path=confined(in.at(key).get<std::string>());
    if(fs::path(path).extension()!=".npz")invalid(std::string(key)+" must be a .npz table","BAD_REF");
    S8Table t;t.bytes=frames::bounded_read(path);t.meta=frames::npz(t.bytes);return t;
}
std::string s8_dtype(const S8Table& t,const std::string& column){return t.meta["column_meta"][column]["dtype"].get<std::string>();}
bool s8_has(const S8Table& t,const std::string& column){for(const auto& c:t.meta["columns"])if(c==column)return true;return false;}
void s8_need_column(const S8Table& t,const std::string& column){if(!s8_has(t,column))invalid("unknown column: "+column,"UNKNOWN_COLUMN");}
std::vector<std::string> s8_names(const Json& v,const char* what,size_t maximum=S8_COLS,bool allow_empty=false){
    if(!v.is_array())bad_param(std::string(what)+" must be a list of column names");
    if(v.empty()&&!allow_empty)invalid(std::string(what)+" must name at least one column","BAD_DATA");
    if(v.size()>maximum)bad_param(std::string(what)+" lists too many columns");
    std::vector<std::string> out;std::set<std::string> seen;
    for(const auto& x:v){if(!x.is_string())bad_param(std::string(what)+" entries must be strings");std::string s=x;if(!seen.insert(s).second)invalid(std::string("duplicate column in ")+what+": "+s,"DUPLICATE_COLUMN");out.push_back(s);}
    return out;
}
bool s8_numeric(const std::string& dtype){return dtype=="int64"||dtype=="float64"||dtype=="bool";}
void s8_table_out(const Params& in,S8Extra& x){
    auto out=confined(in.at("out").get<std::string>(),true);
    if(fs::path(out).extension()!=".npz")invalid("out must end in .npz","BAD_REF");
    x.table_out=out;
}
uint64_t s8_rows(const S8Table& t){return t.meta["n_rows"].get<uint64_t>();}
bool s8_identifier(const std::string& name){
    try{check_variable_name(name,"column");return true;}catch(const Error&){return false;}
}

struct S8Member{size_t offset;uint64_t width,rows;std::string dtype;};
S8Member s8_member(const S8Table& t,const std::string& name){
    const auto& b=t.bytes;size_t p=frames::le(b,b.size()-6,4);auto count=frames::le(b,b.size()-12,2);
    for(size_t i=0;i<count;++i){
        auto nl=frames::le(b,p+28,2),local=frames::le(b,p+42,4);
        if(b.substr(p+46,nl)==name+".npy"){
            size_t data=local+30+nl+frames::le(b,local+28,2),h=frames::le(b,data+8,2);
            auto parsed=parse_npy_header(b.substr(data+10,h));auto bytes=frames::le(b,p+24,4)-10-h;
            uint64_t n=parsed.shape.at(0),width=n?bytes/n:0;
            return {data+10+h,width,n,parsed.descr};
        }
        p+=46+nl;
    }
    invalid("missing table member","BAD_DATA");throw std::runtime_error("unreachable");
}
void s8_classes(const S8Table& t,const std::vector<std::string>& columns){
    std::set<long double> numbers;std::set<std::string> strings;
    for(const auto& c:columns){
        size_t i=0;while(t.meta["columns"][i]!=c)++i;
        auto raw=s8_member(t,"c"+std::to_string(i)),mask=s8_member(t,"m"+std::to_string(i));
        auto dtype=s8_dtype(t,c);uint64_t n=s8_rows(t);
        if(raw.rows!=n||mask.rows!=n||mask.dtype!="|b1")invalid("table column shape/mask mismatch","BAD_DATA");
        if((dtype=="int64"||dtype=="category")&&raw.dtype!="<i8")invalid("table dtype mismatch","BAD_DATA");
        if(dtype=="float64"&&raw.dtype!="<f8")invalid("table dtype mismatch","BAD_DATA");
        if(dtype=="bool"&&raw.dtype!="|b1")invalid("table dtype mismatch","BAD_DATA");
        if(dtype=="string"&&raw.dtype.rfind("|S",0)!=0)invalid("table dtype mismatch","BAD_DATA");
        for(size_t r=0;r<n;++r){
            if(t.bytes[mask.offset+r])continue;
            const char* value=t.bytes.data()+raw.offset+r*raw.width;
            if(dtype=="string"){std::string v(value,raw.width);auto z=v.find('\0');if(z!=std::string::npos)v.resize(z);strings.insert(v);}
            else if(dtype=="category"){int64_t v;std::memcpy(&v,value,8);const auto& levels=t.meta["column_meta"][c]["categories"];if(v<0||uint64_t(v)>=levels.size())invalid("invalid category code","BAD_DATA");strings.insert(levels[v].get<std::string>());}
            else if(dtype=="float64"){double v;std::memcpy(&v,value,8);if(std::isfinite(v))numbers.insert(v);}
            else if(dtype=="bool")numbers.insert(*value!=0);
            else{int64_t v;std::memcpy(&v,value,8);numbers.insert(v);}
            if(numbers.size()+strings.size()>S8_CLASSES)invalid("classification metrics allow at most 30 distinct labels","INPUT_TOO_LARGE");
        }
    }
}

// ---- bridge ----
Json prepare_to_array(const Params& in,bool twin,S8Extra& x){
    Json p=in;S8Table t=s8_table(in,"frame");
    auto columns=s8_names(in.at("columns"),"columns");
    for(const auto& c:columns){s8_need_column(t,c);const std::string d=s8_dtype(t,c);if(!s8_numeric(d))invalid("column "+c+" has dtype "+d+": only int64, float64 and bool convert to an array","BAD_DTYPE");}
    if(in["squeeze"]&&columns.size()!=1)bad_param("squeeze needs exactly one column");
    int64_t offset=in["offset"];if(offset<0)bad_param("offset must be >= 0");
    const uint64_t n=s8_rows(t);
    if(uint64_t(offset)>=n)invalid("offset "+std::to_string(offset)+" selects no rows (the table has "+std::to_string(n)+")","BAD_DATA");
    uint64_t rows=n-uint64_t(offset);
    if(in.contains("limit")&&!in["limit"].is_null()){int64_t limit=in["limit"];if(limit<1)bad_param("limit must be >= 1");rows=std::min<uint64_t>(rows,uint64_t(limit));}
    const uint64_t values=rows*columns.size();
    if(values>S8_VALUES)invalid("the array would hold "+std::to_string(values)+" values (limit "+std::to_string(S8_VALUES)+"): select fewer rows or columns","INPUT_TOO_LARGE");
    if(!twin&&values>S8_INLINE)output_too_large("the array would hold "+std::to_string(values)+" values inline (limit 1000); use frame_to_array_to");
    if(twin){std::string key=in["key"];check_key(key);if(key=="__metadata__")invalid("reserved key","BAD_KEY");p["out"]=output_path(in);x.tensor_write=true;}
    x.tables["frame"]=std::move(t.bytes);return p;
}

Json prepare_array_to_frame(const Params& in,std::map<std::string,Array>& arrays,S8Extra& x){
    Json p=in;Array a=load(in.at("ref"));
    if(a.shape.size()==1)a.shape.push_back(1);
    if(a.shape.size()!=2)invalid("ref must be a vector or a matrix","BAD_DATA");
    const size_t rows=size_t(a.shape[0]),cols=size_t(a.shape[1]);  // a vector was just given the shape [n, 1]
    if(cols>S8_COLS)invalid("a table holds at most 256 columns","INPUT_TOO_LARGE");
    if(rows*cols>S8_CELLS||rows>200000)invalid("a table holds at most 200000 rows and 500000 cells","INPUT_TOO_LARGE");
    auto columns=s8_names(in.at("columns"),"columns");
    if(columns.size()!=cols)invalid("columns names "+std::to_string(columns.size())+" columns but ref has "+std::to_string(cols),"BAD_DATA");
    {Json names=Json::array();for(const auto& c:columns)names.push_back(c);frames::names(names);}
    std::map<std::string,std::string> dtypes;
    if(in.contains("dtypes")&&!in["dtypes"].is_null()){
        if(!in["dtypes"].is_object())bad_param("dtypes must map column names to float64, int64 or bool");
        for(auto it=in["dtypes"].begin();it!=in["dtypes"].end();++it){
            if(std::find(columns.begin(),columns.end(),it.key())==columns.end())invalid("dtypes names an unknown column: "+it.key(),"UNKNOWN_COLUMN");
            if(!it.value().is_string()||(it.value()!="float64"&&it.value()!="int64"&&it.value()!="bool"))bad_param("dtypes entries must be float64, int64 or bool");
            dtypes[it.key()]=it.value();
        }
    }
    const std::string source=a.dtype;const std::string fallback=(source=="I64"||source=="I32"||source=="U8")?"int64":source=="BOOL"?"bool":"float64";
    Json resolved=Json::object();
    for(size_t j=0;j<cols;++j){
        std::string d=dtypes.count(columns[j])?dtypes[columns[j]]:fallback;resolved[columns[j]]=d;
        if(d=="float64")continue;  // NaN / Inf are legal float64 values (masked only with nan_policy=missing)
        for(size_t i=0;i<rows;++i){
            double v=a.data[i*cols+j];
            if(!std::isfinite(v))invalid("column "+columns[j]+" is "+d+" but contains NaN/Inf at row "+std::to_string(i),"BAD_DATA");
            if(d=="bool"&&v!=0&&v!=1)invalid("column "+columns[j]+" is bool but row "+std::to_string(i)+" is not exactly 0 or 1","BAD_DATA");
            if(d=="int64"&&(v!=std::floor(v)||std::fabs(v)>S8_EXACT))invalid("column "+columns[j]+" is int64 but row "+std::to_string(i)+" is not an exactly representable integer","BAD_DATA");
        }
    }
    if(in.contains("frame")&&!in["frame"].is_null()){
        S8Table t=s8_table(in,"frame");
        if(s8_rows(t)!=rows)invalid("frame and ref need identical row counts","BAD_DATA");
        Json names=t.meta["columns"];
        for(const auto& c:columns){
            if(s8_has(t,c)){if(!in["allow_replace"].get<bool>())bad_param("column "+c+" already exists: set allow_replace to overwrite it");}
            else names.push_back(c);
        }
        frames::names(names);if(rows*names.size()>S8_CELLS)invalid("enriched table exceeds cell cap","INPUT_TOO_LARGE");
        uint64_t bytes=t.bytes.size()+rows*cols*9+1024*1024;
        if(bytes>frames::BYTES)output_too_large("enriched table exceeds 256MiB storage budget");
        x.tables["frame"]=std::move(t.bytes);
    }
    p["dtypes"]=resolved;p["rows"]=rows;
    s8_table_out(in,x);arrays["ref"]=std::move(a);return p;
}

// ---- mutate ----
Json prepare_mutate(const Params& in,S8Extra& x){
    Json p=in;S8Table t=s8_table(in,"frame");
    if(!in["assign"].is_array()||in["assign"].empty()||in["assign"].size()>S8_ASSIGN)bad_param("assign needs 1..32 {name, expr} entries");
    const bool replace=in["allow_replace"];
    std::set<std::string> available;std::set<std::string> existing;
    for(const auto& c:t.meta["columns"]){std::string name=c;existing.insert(name);if(s8_numeric(s8_dtype(t,name))&&s8_identifier(name))available.insert(name);}
    std::set<std::string> assigned;Json plan=Json::array();size_t columns=existing.size();
    for(const auto& item:in["assign"]){
        if(!item.is_object()||item.size()!=2||!item.contains("name")||!item.contains("expr")||!item["name"].is_string()||!item["expr"].is_string())bad_param("each assign entry is exactly {name, expr}");
        std::string name=item["name"];{Json one=Json::array({name});frames::names(one);}
        if(!assigned.insert(name).second)invalid("column assigned twice: "+name,"DUPLICATE_COLUMN");
        const bool replaced=existing.count(name)>0;
        if(replaced&&!replace)bad_param("column "+name+" already exists: set allow_replace to overwrite it");
        std::set<std::string> refs;
        Json ast=parse_expression(item["expr"].get<std::string>(),available,{},false,"assign."+name,&refs);
        plan.push_back({{"name",name},{"ast",ast},{"refs",refs},{"replaced",replaced}});
        if(!replaced)++columns;
        if(s8_identifier(name))available.insert(name);
    }
    if(columns>S8_COLS)invalid("the result would hold "+std::to_string(columns)+" columns (limit 256)","TOO_MANY_COLUMNS");
    if(s8_rows(t)*columns>S8_CELLS)invalid("the result would hold "+std::to_string(s8_rows(t)*columns)+" cells (limit 500000)","INPUT_TOO_LARGE");
    const std::string policy=in["nonfinite"];if(policy!="error"&&policy!="mask"&&policy!="keep")bad_param("nonfinite must be error, mask or keep");
    p["plan"]=plan;p.erase("assign");x.needs_spec=true;s8_table_out(in,x);x.tables["frame"]=std::move(t.bytes);return p;
}

// ---- merge ----
std::string s8_key_class(const S8Table& t,const std::string& column){
    const std::string d=s8_dtype(t,column);
    if(d=="float64")invalid("column "+column+" is float64: floating-point keys are not joined","BAD_DTYPE");
    if(d=="string"||d=="category")return "label";
    if(d=="datetime64[ns]")return "datetime:"+t.meta["column_meta"][column].value("tz",std::string());
    return d;
}
Json prepare_merge(const Params& in,S8Extra& x){
    Json p=in;S8Table l=s8_table(in,"left"),r=s8_table(in,"right");
    const bool has_on=in.contains("on")&&!in["on"].is_null(),has_l=in.contains("left_on")&&!in["left_on"].is_null(),has_r=in.contains("right_on")&&!in["right_on"].is_null();
    if(has_on==(has_l||has_r))bad_param("give either on, or left_on together with right_on");
    if(!has_on&&!(has_l&&has_r))bad_param("left_on and right_on must be given together");
    std::vector<std::string> lk,rk;
    if(has_on){lk=rk=s8_names(in["on"],"on",8);}
    else{lk=s8_names(in["left_on"],"left_on",8);rk=s8_names(in["right_on"],"right_on",8);if(lk.size()!=rk.size())bad_param("left_on and right_on need the same number of columns");}
    for(size_t i=0;i<lk.size();++i){
        s8_need_column(l,lk[i]);s8_need_column(r,rk[i]);
        if(s8_key_class(l,lk[i])!=s8_key_class(r,rk[i]))invalid("key "+lk[i]+" ("+s8_dtype(l,lk[i])+") and "+rk[i]+" ("+s8_dtype(r,rk[i])+") have incompatible types","KEY_DTYPE_MISMATCH");
    }
    const Json& suf=in.at("suffixes");
    if(!suf.is_array()||suf.size()!=2||!suf[0].is_string()||!suf[1].is_string())bad_param("suffixes needs two strings");
    std::string sl=suf[0],sr=suf[1];if(sl.empty()||sr.empty()||sl==sr)bad_param("suffixes must be two different non-empty strings");
    // the output naming plan, identical to the worker's: colliding non-key names get a suffix on both sides
    std::vector<std::string> lcols,rcols;for(const auto& c:l.meta["columns"])lcols.push_back(c);for(const auto& c:r.meta["columns"])rcols.push_back(c);
    std::set<std::string> rdrop;if(has_on)for(const auto& k:rk)rdrop.insert(k);
    std::set<std::string> lset(lcols.begin(),lcols.end());std::vector<std::string> rkept;
    for(const auto& c:rcols)if(!rdrop.count(c))rkept.push_back(c);
    std::set<std::string> clash;for(const auto& c:rkept)if(lset.count(c)&&!(has_on&&std::find(lk.begin(),lk.end(),c)!=lk.end()))clash.insert(c);
    std::vector<std::string> names;for(const auto& c:lcols)names.push_back(clash.count(c)?c+sl:c);for(const auto& c:rkept)names.push_back(clash.count(c)?c+sr:c);
    std::set<std::string> unique(names.begin(),names.end());
    if(unique.size()!=names.size())invalid("the merged table would hold duplicate column names; choose different suffixes","DUPLICATE_COLUMN");
    frames::names(names);
    if(names.size()>S8_COLS)invalid("the merged table would hold "+std::to_string(names.size())+" columns (limit 256)","TOO_MANY_COLUMNS");
    int64_t max_rows=in["max_rows"];if(max_rows<1||max_rows>200000)bad_param("max_rows must be 1..200000");
    const std::string how=in["how"];if(how!="inner"&&how!="left"&&how!="right"&&how!="outer")bad_param("how must be inner, left, right or outer");
    const std::string validate=in["validate"];if(validate!="one_to_one"&&validate!="one_to_many"&&validate!="many_to_one"&&validate!="many_to_many")bad_param("validate must be one_to_one, one_to_many, many_to_one or many_to_many");
    p["left_keys"]=lk;p["right_keys"]=rk;p["key_shared"]=has_on;p["output_names"]=names;p.erase("on");p.erase("left_on");p.erase("right_on");
    s8_table_out(in,x);x.tables["left"]=std::move(l.bytes);x.tables["right"]=std::move(r.bytes);return p;
}

// ---- rolling ----
Json prepare_rolling(const Params& in,S8Extra& x){
    Json p=in;S8Table t=s8_table(in,"frame");
    auto columns=s8_names(in.at("columns"),"columns",64);
    for(const auto& c:columns){s8_need_column(t,c);const std::string d=s8_dtype(t,c);if(!s8_numeric(d))invalid("column "+c+" has dtype "+d+": rolling windows need int64, float64 or bool","BAD_DTYPE");}
    static const std::set<std::string> known={"sum","mean","std","var","min","max","median","count"};
    if(!in["functions"].is_array()||in["functions"].empty()||in["functions"].size()>8)bad_param("functions needs 1..8 of sum, mean, std, var, min, max, median, count");
    std::vector<std::string> functions;std::set<std::string> seen;
    for(const auto& f:in["functions"]){if(!f.is_string()||!known.count(f.get<std::string>()))bad_param("unknown rolling function");if(!seen.insert(f.get<std::string>()).second)bad_param("duplicate rolling function");functions.push_back(f);}
    const bool has_w=in.contains("window")&&!in["window"].is_null(),has_t=in.contains("time_window")&&!in["time_window"].is_null();
    if(has_w==has_t)bad_param("give exactly one of window (rows) and time_window");
    const uint64_t rows=s8_rows(t);
    uint64_t work=0;
    if(has_w){
        int64_t window=in["window"];if(window<1||window>int64_t(S8_WINDOW))bad_param("window must be 1..100000 rows");
        if(in.contains("on")&&!in["on"].is_null())bad_param("on is only used with time_window");
        int64_t mp=in.contains("min_periods")&&!in["min_periods"].is_null()?in["min_periods"].get<int64_t>():window;
        if(mp<1||mp>window)bad_param("min_periods must be 1..window");
        p["min_periods"]=mp;work=rows*std::min<uint64_t>(uint64_t(window),rows)*columns.size()*functions.size();
        if(work>S8_WORK)invalid("the rolling work (rows x window x columns x functions = "+std::to_string(work)+") exceeds "+std::to_string(S8_WORK),"INPUT_TOO_LARGE");
    }else{
        if(!in.contains("on")||in["on"].is_null())bad_param("time_window needs the datetime column in on");
        if(in["center"])bad_param("center is not available with time_window");
        std::string on=in["on"];s8_need_column(t,on);if(s8_dtype(t,on)!="datetime64[ns]")invalid("on must be a datetime64[ns] column","BAD_DTYPE");
        std::string w=in["time_window"];size_t i=0;while(i<w.size()&&w[i]>='0'&&w[i]<='9')++i;
        static const std::map<std::string,double> unit={{"s",1.0},{"min",60.0},{"h",3600.0},{"D",86400.0}};
        if(!i||i>7||w[0]=='0'||!unit.count(w.substr(i)))bad_param("time_window is <count><unit> with unit s, min, h or D (fixed-length), e.g. 30s");
        p["time_window_seconds"]=std::stod(w.substr(0,i))*unit.at(w.substr(i));
        int64_t mp=in.contains("min_periods")&&!in["min_periods"].is_null()?in["min_periods"].get<int64_t>():1;if(mp<1||mp>S8_WINDOW)bad_param("min_periods must be >= 1");p["min_periods"]=mp;
    }
    std::set<std::string> existing;for(const auto& c:t.meta["columns"])existing.insert(c);size_t total=existing.size();
    for(const auto& c:columns)for(const auto& f:functions){if(!existing.insert(c+"__"+f).second)invalid("output column "+c+"__"+f+" already exists","DUPLICATE_COLUMN");++total;}
    frames::names(Json(existing));
    if(total>S8_COLS)invalid("the result would hold "+std::to_string(total)+" columns (limit 256)","TOO_MANY_COLUMNS");
    if(rows*total>S8_CELLS)invalid("the result would hold "+std::to_string(rows*total)+" cells (limit 500000)","INPUT_TOO_LARGE");
    p["functions"]=functions;s8_table_out(in,x);x.tables["frame"]=std::move(t.bytes);return p;
}

// ---- metrics ----
Json prepare_metrics(const Params& in,std::map<std::string,Array>& arrays,S8Extra& x){
    Json p=in;
    static const std::set<std::string> classification={"accuracy","balanced_accuracy","precision","recall","f1","confusion_matrix"},scored={"roc_auc","log_loss","brier"},regression={"r2","mse","mae","mape"};
    if(!in["metrics"].is_array()||in["metrics"].empty()||in["metrics"].size()>14)bad_param("metrics needs 1..14 metric names");
    std::set<std::string> wanted;for(const auto& m:in["metrics"]){
        if(!m.is_string()||!(classification.count(m.get<std::string>())||scored.count(m.get<std::string>())||regression.count(m.get<std::string>())))bad_param("unknown metric");
        if(!wanted.insert(m.get<std::string>()).second)bad_param("duplicate metric");
    }
    auto any=[&](const std::set<std::string>& group){for(const auto& m:wanted)if(group.count(m))return true;return false;};
    const bool need_pred=any(classification)||any(regression),need_score=any(scored);
    auto given=[&](const char* k){return in.contains(k)&&!in[k].is_null();};
    if(need_pred&&!given("y_pred"))bad_param("these metrics need y_pred");
    if(need_score&&!given("y_score"))bad_param("roc_auc, log_loss and brier need y_score");
    if(!given("y_true"))bad_param("y_true is required");
    const std::string average=in["average"];if(average!="binary"&&average!="macro"&&average!="micro"&&average!="weighted")bad_param("average must be binary, macro, micro or weighted");
    const std::string zd=in["zero_division"];if(zd!="undefined"&&zd!="0"&&zd!="1")bad_param("zero_division must be undefined, 0 or 1");
    const bool prf_binary=(wanted.count("precision")||wanted.count("recall")||wanted.count("f1"))&&average=="binary";
    if(given("pos_label")&&!(prf_binary||any(scored)))bad_param("pos_label is only used by binary precision/recall/f1, roc_auc, log_loss and brier");
    if(given("pos_label")&&!(in["pos_label"].is_string()||in["pos_label"].is_boolean()||(in["pos_label"].is_number()&&std::isfinite(in["pos_label"].get<double>()))))bad_param("pos_label must be a finite scalar label");
    const bool framed=given("frame");
    if(framed){
        S8Table t=s8_table(in,"frame");
        for(const char* k:{"y_true","y_pred","y_score"})if(given(k)){
            if(!in[k].is_string())bad_param(std::string(k)+" must name a column when frame is given");
            std::string c=in[k];s8_need_column(t,c);const std::string d=s8_dtype(t,c);
            if(d=="datetime64[ns]")invalid("column "+c+" is a datetime","BAD_DTYPE");
            if((std::string(k)=="y_score"||any(regression))&&!s8_numeric(d))invalid("column "+c+" has dtype "+d+" but a numeric score/target is needed","BAD_DTYPE");
        }
        if(s8_rows(t)<1)invalid("the table has no rows","BAD_DATA");
        if(any(classification)){std::vector<std::string> cols={in["y_true"].get<std::string>(),in["y_pred"].get<std::string>()};s8_classes(t,cols);}
        x.tables["frame"]=std::move(t.bytes);
    }else{
        size_t n=0;
        for(const char* k:{"y_true","y_pred","y_score"})if(given(k)){
            Array a=load(in[k]);if(a.shape.size()!=1)invalid(std::string(k)+" must be a vector","BAD_DATA");need_finite(a,k);
            if(n&&a.data.size()!=n)invalid("y_true, y_pred and y_score must have the same length","BAD_DATA");n=a.data.size();
            if(std::string(k)=="y_score"&&(wanted.count("log_loss")||wanted.count("brier")))for(double v:a.data)if(v<0||v>1)invalid("y_score must be a probability in [0, 1] for log_loss and brier","BAD_DATA");
            arrays[k]=std::move(a);
        }
        if(any(classification)){
            std::set<double> labels;for(const char* k:{"y_true","y_pred"})if(arrays.count(k))for(double v:arrays[k].data)labels.insert(v);
            if(labels.size()>S8_CLASSES)invalid("classification metrics allow at most 30 distinct labels (found "+std::to_string(labels.size())+")","INPUT_TOO_LARGE");
        }
    }
    return p;
}

// ---- sampling ----
Json prepare_rvs(const std::string& op,const Params& in,S8Extra& x){
    Json p=in;const bool twin=s7_twin(op);
    if(in["fn"]!="rvs")bad_param("the write form of stats_dist only samples: use fn rvs");
    if(in.contains("x")&&!in["x"].is_null())bad_param("x is only used by pdf, cdf, ppf and sf");
    if(!in.contains("seed")||in["seed"].is_null())bad_param("rvs needs an explicit seed (non-negative integer below 2^63)");
    if(!in["seed"].is_number_integer())bad_param("seed must be an integer");
    if(in["seed"].is_number_unsigned()){if(in["seed"].get<uint64_t>()>uint64_t(INT64_MAX))bad_param("seed must be below 2^63");}
    else if(in["seed"].get<int64_t>()<0)bad_param("seed must be >= 0");
    if(!in.contains("n")||in["n"].is_null())bad_param("rvs needs the sample size n");
    int64_t n=in["n"];if(n<1)bad_param("n must be >= 1");
    if(twin){if(size_t(n)>S8_RVS_FILE)invalid("n is limited to "+std::to_string(S8_RVS_FILE)+" values in a file","INPUT_TOO_LARGE");}
    else if(size_t(n)>S8_RVS_INLINE)output_too_large("n="+std::to_string(n)+" exceeds the 1000 inline values; use stats_dist_to");
    const std::string dist=in["dist"];const Json& prm=in.at("params");
    if(!prm.is_object())bad_param("params must be an object");
    static const std::map<std::string,std::vector<std::pair<std::string,double>>> table={
        {"norm",{{"loc",0.0},{"scale",1.0}}},{"t",{{"df",NAN},{"loc",0.0},{"scale",1.0}}},{"uniform",{{"low",0.0},{"high",1.0}}},{"expon",{{"scale",1.0}}},
        {"gamma",{{"shape",NAN},{"scale",NAN}}},{"beta",{{"a",NAN},{"b",NAN}}},{"lognorm",{{"mu",0.0},{"sigma",1.0}}},
        {"chi2",{{"df",NAN}}},{"poisson",{{"lam",NAN}}},{"binom",{{"n",NAN},{"p",NAN}}}};
    auto spec=table.find(dist);if(spec==table.end())bad_param("unknown distribution");
    std::set<std::string> allowed;for(const auto& k:spec->second)allowed.insert(k.first);
    for(auto it=prm.begin();it!=prm.end();++it)if(!allowed.count(it.key()))bad_param("parameter "+it.key()+" is not used by "+dist);
    std::map<std::string,double> v;
    for(const auto& k:spec->second){
        if(prm.contains(k.first)){if(!prm[k.first].is_number()||!std::isfinite(prm[k.first].get<double>()))bad_param(k.first+" must be a finite number");v[k.first]=prm[k.first];}
        else if(!std::isnan(k.second))v[k.first]=k.second;else bad_param(dist+" needs parameter "+k.first);
        if(std::fabs(v[k.first])>1e12)bad_param(k.first+" is out of range");
    }
    auto positive=[&](const char* k){if(!(v[k]>0))bad_param(std::string(k)+" must be > 0");};
    if(dist=="norm"||dist=="expon")positive("scale");
    if(dist=="t"){positive("df");positive("scale");}
    if(dist=="uniform"&&!(v["low"]<v["high"]))bad_param("low must be below high");
    if(dist=="gamma"){positive("shape");positive("scale");}
    if(dist=="beta"){positive("a");positive("b");}
    if(dist=="lognorm")positive("sigma");
    if(dist=="chi2")positive("df");
    if(dist=="poisson"&&(v["lam"]<0||v["lam"]>1e9))bad_param("lam must be in [0, 1e9]");
    if(dist=="binom"){if(v["n"]<1||v["n"]>1e9||v["n"]!=std::floor(v["n"]))bad_param("n must be an integer in [1, 1e9]");if(v["p"]<0||v["p"]>1)bad_param("p must be in [0, 1]");}
    Json resolved=Json::object();for(const auto& e:v)resolved[e.first]=e.second;p["params"]=resolved;
    if(twin){std::string key=in["key"];check_key(key);if(key=="__metadata__")invalid("reserved key","BAD_KEY");p["out"]=output_path(in);x.tensor_write=true;}
    return p;
}

Json prepare_s8(const std::string& op,const Params& in,std::map<std::string,Array>& arrays,S8Extra& x){
    if(op=="frame_to_array"||op=="frame_to_array_to")return prepare_to_array(in,op=="frame_to_array_to",x);
    if(op=="array_to_frame")return prepare_array_to_frame(in,arrays,x);
    if(op=="frame_mutate")return prepare_mutate(in,x);
    if(op=="frame_merge")return prepare_merge(in,x);
    if(op=="frame_rolling")return prepare_rolling(in,x);
    if(op=="ml_metrics")return prepare_metrics(in,arrays,x);
    return prepare_rvs(op,in,x);
}

// ---- registration ----
void add_s8_ops(CliSpec& spec){
    auto str=[](const char* name,bool required,Json def,std::vector<Json> values,const char* text){return Param{name,Type::String,required,std::move(def),std::move(values),text};};
    auto strings=[](const char* name,bool required,const char* text){return Param{name,Type::Array,required,nullptr,{},text,"",false,false,false,{},Type::String};};
    auto integer=[](const char* name,Json def,const char* text){return Param{name,Type::Integer,false,std::move(def),{},text};};
    auto json=[](const char* name,bool required,const char* text){return Param{name,Type::Json,required,nullptr,{},text};};
    const char* table_out="New .npz table artifact inside the working directory (CONFLICT if it exists).";
    const char* tensor_out="New .safetensors path inside the working directory (CONFLICT if it exists).";
    const char* limits=" Tables: <=200000 rows, <=256 columns, <=500000 cells. No pandas is imported.";
    for(bool write:{false,true}){
        std::vector<Param> params={
            str("frame",true,nullptr,{},"Existing .npz table (frame_import / any table operation)."),
            strings("columns",true,"Ordered unique columns to convert; each must be int64, float64 or bool (bool maps to 0/1). Strings, categories and datetimes are refused (BAD_DTYPE): encode them first."),
            str("missing",false,"error",{"error","nan"},"error (default): masked missing values and numeric NaN/Inf are NA_NOT_ALLOWED; output is guaranteed finite. nan: masked values become NaN and the result reports missing_mask_lost=true plus both source counts (masked NA and numeric NaN become indistinguishable)."),
            integer("offset",0,"First row to convert (0-based, default 0); selecting no rows is BAD_DATA."),
            integer("limit",nullptr,"Maximum rows to convert from offset, >=1 (default: all)."),
            Param{"squeeze",Type::Boolean,false,false,{},"Exactly one column: return/store a vector of shape [rows] instead of a [rows, 1] matrix (the form sig_*, interp_* and spatial_* operations take)."},
            Param{"allow_lossy_int",Type::Boolean,false,false,{},"Convert int64 values beyond +-2^53 (not exactly representable in binary64) with rounding, counted in lossy_int_count; off by default (INT64_NOT_EXACT)."}};
        if(write){params.push_back(str("key",false,"x",{},"Tensor name in the artifact: [A-Za-z_][A-Za-z0-9_.-]{0,63}."));params.push_back(str("out",true,nullptr,{},tensor_out));}
        spec.operations.push_back(Operation{write?"frame_to_array_to":"frame_to_array",write?Access::Write:Access::Read,
            std::string("Table to rank-2 row-major F64 array (columns in the requested order). Result: shape, columns, source_dtypes, source_na_count, source_nan_count, missing_mask_lost, n_rows_total")+(write?"; the array is stored under artifacts[key] (<=200000 values)":", values (<=1000 values inline; use frame_to_array_to for more)")+"."+limits,"object",params});
    }
    spec.operations.push_back(Operation{"array_to_frame",Access::Write,std::string("Array (inline, .npy or safetensors#key) to a table: explicit unique column names for every column (a vector is one column), dtypes float64 / int64 / bool (default from the source dtype; bool needs exactly 0/1, int64 exact integers), no implicit index. NaN stays a numeric value unless nan_policy=missing. Result: shape, column_meta, na_counts, nan_count, preview, artifacts.table.")+limits,"object",{
        str("frame",false,nullptr,{},"Optional source table: append ref columns by row position, requiring identical row counts; preserve source dtypes, masks and order."),
        Param{"allow_replace",Type::Boolean,false,false,{},"With frame, explicitly allow replacing existing named columns (default false)."},
        json("ref",true,"Finite or NaN/Inf-containing vector or matrix, <=200000 numbers: inline JSON, file.npy or file.safetensors#key. Rows are table rows."),
        strings("columns",true,"One unique non-empty name (<=256 bytes UTF-8) per column of ref."),
        Param{"dtypes",Type::Object,false,nullptr,{},"Optional {column: float64|int64|bool} overrides; file sources I64/I32/U8 -> int64, BOOL -> bool; inline JSON numbers always arrive as float64. Set explicit overrides for inline integers."},
        str("nan_policy",false,"value",{"value","missing"},"value (default): NaN stays a float64 value. missing: NaN becomes a masked missing value (float64 columns only)."),
        str("out",true,nullptr,{},table_out)}});
    spec.operations.push_back(Operation{"frame_mutate",Access::Write,std::string("New or replaced float64 columns from expressions in the existing safe grammar (arithmetic, ifelse with comparisons and & | only inside its condition, math functions; no reducers, no indexing, no eval path). Columns are int64/float64/bool with identifier-style names ([a-z][a-z0-9_]{0,31}); assignments run in order and may use earlier ones. A masked missing value in any referenced column makes the result missing. Bare comparison/logic expressions are EXPR_SYNTAX; use ifelse(condition,1,0). NaN conditions produce NaN. nonfinite decides what to do with NaN/Inf results (error, mask or keep). Result: shape, assigned[{name, replaced, n_masked, n_nonfinite}], column_meta, preview, artifacts.table.")+limits,"object",{
        str("frame",true,nullptr,{},"Existing .npz table."),
        Param{"assign",Type::Array,true,nullptr,{},"1..32 entries {name, expr}; expr <=2048 bytes / 512 nodes.","",false,false,false,{},Type::Object},
        Param{"allow_replace",Type::Boolean,false,false,{},"Allow an assignment to overwrite an existing column (default false: BAD_PARAM)."},
        str("nonfinite",false,"error",{"error","mask","keep"},"NaN/Inf produced by an expression on non-missing inputs: error (NON_FINITE, default), mask (becomes a masked missing value, counted) or keep (stored as a float64 value)."),
        str("out",true,nullptr,{},table_out)}});
    spec.operations.push_back(Operation{"frame_merge",Access::Write,std::string("Join two tables on explicit keys (inner, left, right, outer) with validation and a row-explosion cap checked from key counts before anything is built. Keys are int64, bool, string/category (compared by label) or datetime columns of matching type, never float64; rows with a missing key never match (they stay unmatched in left/right/outer). Order: inner/left/outer follow the left rows (matches in right row order), right follows the right rows (matches in left row order); outer then appends the unmatched right rows. With on the key is one column (taken from the left, else the right); with left_on/right_on both are kept. Non-key names present on both sides get suffixes on both. Unmatched cells are masked, dtypes are preserved. Result: shape, how, n_pairs, n_left_only, n_right_only, columns, column_meta, preview, artifacts.table.")+limits,"object",{
        str("left",true,nullptr,{},"Left .npz table."),str("right",true,nullptr,{},"Right .npz table."),
        strings("on",false,"Key columns present in both tables (1..8). Give on, or left_on with right_on."),
        strings("left_on",false,"Key columns of the left table (with right_on)."),strings("right_on",false,"Key columns of the right table, same count as left_on."),
        str("how",false,"inner",{"inner","left","right","outer"},"Join type."),
        str("validate",false,"many_to_many",{"one_to_one","one_to_many","many_to_one","many_to_many"},"Key uniqueness contract: one_to_one needs unique keys on both sides, including repeated missing-key rows, one_to_many a unique left, many_to_one a unique right; a violation is MERGE_VALIDATION. many_to_many (default) checks nothing."),
        Param{"suffixes",Type::Array,false,Json::array({"_x","_y"}),{},"Two different non-empty suffixes for colliding non-key names (default _x, _y); the result must still have unique names (DUPLICATE_COLUMN).","",false,false,false,{},Type::String},
        integer("max_rows",200000,"Cap on result rows, 1..200000 (also bounded by the 500000-cell table cap); exceeding it is ROW_EXPLOSION and reports the row count the join would have produced."),
        str("out",true,nullptr,{},table_out)}});
    spec.operations.push_back(Operation{"frame_rolling",Access::Write,std::string("Rolling-window statistics of numeric columns appended as <column>__<function> (original columns kept). Count windows are trailing (window rows ending at the row, or centred with center: rows i-window/2 .. i+(window-1)/2); time windows (time_window, with an explicit sorted non-missing datetime column on) cover (t-window, t] over the rows up to and including the row. Missing values are skipped; a result needs min_periods observations (std/var also two; ddof=1), otherwise it is missing; count is nullable int64 and also needs min_periods non-missing observations. Work (rows x window x columns x functions) is capped at 1e8. Result: shape, columns_added, preview, artifacts.table.")+limits,"object",{
        str("frame",true,nullptr,{},"Existing .npz table."),
        strings("columns",true,"Numeric columns (int64, float64, bool) to roll, <=64."),
        strings("functions",true,"1..8 of sum, mean, std, var, min, max, median, count."),
        integer("window",nullptr,"Rows per window, 1..100000 (give window or time_window)."),
        str("time_window",false,nullptr,{},"Fixed-length time window <count><unit>, unit s, min, h or D (e.g. 30s); needs on."),
        str("on",false,nullptr,{},"Datetime64[ns] column for time_window; must be sorted ascending with no missing values (NOT_SORTED otherwise)."),
        integer("min_periods",nullptr,"Observations needed for a result, 1..window (default: window; time windows: 1)."),
        Param{"center",Type::Boolean,false,false,{},"Count windows only: centre the window on the row."},
        str("out",true,nullptr,{},table_out)}});
    spec.operations.push_back(Operation{"ml_metrics",Access::Read,"Model evaluation metrics computed by hand formulas (numpy only; no scikit-learn import). Classification: accuracy, balanced_accuracy (mean recall over the classes present in y_true), precision, recall, f1 (average binary/macro/micro/weighted), confusion_matrix (rows true, columns predicted, labels sorted; <=30 labels). Scored: roc_auc (rank statistic with ties as 1/2), log_loss (probabilities clipped to [1e-15, 1-1e-15], reported in log_loss_eps) and brier, binary with pos_label. Regression: r2, mse, mae, mape (fraction, not percent). A metric that is mathematically undefined (constant y_true for r2, a zero in y_true for mape, one class for roc_auc, 0/0 in precision/recall with zero_division=undefined) is listed under undefined with a reason, the status is UNDEFINED and the defined metrics are still returned. Data come from a table (frame with column names) or from arrays; NaN/missing values are refused (NA_NOT_ALLOWED / BAD_DATA).","object",{
        Param{"metrics",Type::Array,true,nullptr,{},"1..14 metric names: accuracy, balanced_accuracy, precision, recall, f1, confusion_matrix, roc_auc, log_loss, brier, r2, mse, mae, mape.","",false,false,false,{},Type::String},
        str("frame",false,nullptr,{},"Optional .npz table; then y_true/y_pred/y_score are column names (labels may be int64, bool, string or category)."),
        json("y_true",true,"Column name (with frame) or numeric vector (inline <=1000, .npy, safetensors#key) of true labels/targets."),
        json("y_pred",false,"Column name or vector of predicted labels/values (classification and regression metrics)."),
        json("y_score",false,"Column name or vector of scores / probabilities of pos_label (roc_auc, log_loss, brier); log_loss and brier need values in [0, 1]."),
        str("average",false,"binary",{"binary","macro","micro","weighted"},"Averaging for precision, recall and f1 (default binary, which uses pos_label); see zero_division for undefined denominator handling."),
        json("pos_label",false,"Positive class for binary precision/recall/f1 and for roc_auc, log_loss, brier. Default 1 when the labels are 0/1 (or false/true); required otherwise."),
        str("zero_division",false,"undefined",{"undefined","0","1"},"Value for precision/recall/f1 terms with a zero denominator, or undefined (default: the metric is reported undefined)."),
        }});
}
