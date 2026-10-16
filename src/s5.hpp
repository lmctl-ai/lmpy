// S5 statsmodels front end: validates JSON term lists, families, orders and caps before any process starts.
// No formula string is ever accepted; the worker builds the design matrix itself from the validated term list.
#pragma once
constexpr size_t SM_TERMS=64, SM_NAME_BYTES=256, SM_ARIMA_STATES=64, SM_ARIMA_PARAMS=24, SM_EXOG=8, SM_START=32, SM_INLINE_FORECAST=100, SM_FORECAST=1000, SM_ACF_LAGS=100;
void sm_name(const Json& v,const std::string& what){
    if(!v.is_string())invalid(what+" must be a column name string","BAD_NAME");
    const std::string s=v.get<std::string>();
    if(s.empty()||s.size()>SM_NAME_BYTES)invalid(what+" must be 1..256 bytes","BAD_NAME");
    // Column names are only ever dict keys in the worker and echoed into coefficient names; keep the alphabet free of every character
    // that carries meaning in formulas or in our own naming (col[T.level], a:b, col^2). Non-ASCII UTF-8 letters are allowed.
    for(unsigned char c:s){
        const bool ok=c>=0x80||std::isalnum(c)||c==' '||c=='_'||c=='.'||c=='-'||c=='%'||c=='$'||c=='@'||c=='/'||c=='#';
        if(!ok)invalid(what+" contains a character that is not allowed in a column name reference (letters, digits, space and _ . - % $ @ / # only)","BAD_NAME");
    }
    frames::utf8(s);
}
[[noreturn]] void bad_terms(const std::string& message){invalid(message,"BAD_TERMS");}
// terms := [{"const":bool} | {"var":col} | {"cat":col,"ref":level} | {"interact":[col,col(,col)]} | {"poly":[col,degree 2..6]}]
void sm_terms(const Json& terms){
    if(!terms.is_array()||terms.empty()||terms.size()>SM_TERMS)bad_terms("terms must be a JSON list of 1..64 term objects (formula strings are not accepted)");
    bool constant=false;std::set<std::string> main_columns,interactions;
    for(const auto& term:terms){
        if(!term.is_object())bad_terms("every term is an object with one of the keys const, var, cat, interact, poly");
        size_t primary=0;for(const char* key:{"const","var","cat","interact","poly"})primary+=term.contains(key);
        if(primary!=1)bad_terms("a term has exactly one of the keys const, var, cat, interact, poly");
        const bool cat=term.contains("cat");
        if(term.size()!=(cat?2u:1u))bad_terms("a term has exactly one key (cat also needs ref)");
        if(term.contains("const")){
            if(!term["const"].is_boolean())bad_terms("const must be true or false");
            if(constant)bad_terms("const may appear only once");constant=true;
        }else if(term.contains("var")){
            sm_name(term["var"],"var");if(!main_columns.insert(term["var"].get<std::string>()).second)bad_terms("a column may appear in only one var/cat/poly term");
        }else if(cat){
            if(!term.contains("ref")||!term["ref"].is_string())bad_terms("cat terms need an explicit string ref (the reference level)");
            sm_name(term["cat"],"cat");if(!main_columns.insert(term["cat"].get<std::string>()).second)bad_terms("a column may appear in only one var/cat/poly term");
            if(term["ref"].get<std::string>().size()>SM_NAME_BYTES)bad_terms("ref is too long");
        }else if(term.contains("poly")){
            const Json& p=term["poly"];
            if(!p.is_array()||p.size()!=2||!p[1].is_number_integer()||p[1].get<int64_t>()<2||p[1].get<int64_t>()>6)bad_terms("poly needs [column, integer degree 2..6]");
            sm_name(p[0],"poly");if(!main_columns.insert(p[0].get<std::string>()).second)bad_terms("a column may appear in only one var/cat/poly term");
        }else{
            const Json& f=term["interact"];
            if(!f.is_array()||f.size()<2||f.size()>3)bad_terms("interact needs 2 or 3 column names");
            std::set<std::string> unique;std::string key;
            for(const auto& name:f){sm_name(name,"interact");if(!unique.insert(name.get<std::string>()).second)bad_terms("interact factors must be distinct");}
            for(const auto& name:unique)key+=name+"\x1f";
            if(!interactions.insert(key).second)bad_terms("duplicate interact term");
        }
    }
}
void sm_check_include(const Params& in){
    if(!in.contains("include"))return;std::set<std::string> seen;
    for(const auto& item:in["include"])if(!seen.insert(item.get<std::string>()).second)bad_param("duplicate entry in include");
}
void sm_check_alpha(const Params& in){double a=in["alpha"];if(!(a>0&&a<1))bad_param("alpha must lie strictly between 0 and 1");}
Json prepare_sm(const std::string& op,const Params& input){
    Json p=input;const bool twin=op.size()>3&&op.compare(op.size()-3,3,"_to")==0;
    if(op!="sm_arima"&&op!="sm_arima_to"&&op!="sm_tsa_acf"&&op!="sm_tsa_tools"){
        sm_name(input.at("y"),"y");sm_terms(input.at("terms"));sm_check_include(input);sm_check_alpha(input);
        for(const char* key:{"weights","cluster","offset","exposure"})if(input.contains(key))sm_name(input[key],key);
        const std::string cov=input["cov_type"];
        if(cov=="cluster"&&!input.contains("cluster"))bad_param("cov_type cluster needs the cluster column name");
        if(cov!="cluster"&&input.contains("cluster"))bad_param("cluster is only used with cov_type cluster");
        if(op.rfind("sm_glm",0)==0){
            const std::string family=input["family"];std::string link=input.contains("link")?input["link"].get<std::string>():"";
            static const std::map<std::string,std::vector<std::string>> allowed={{"gaussian",{"identity","log","inverse_power"}},{"poisson",{"log","identity","sqrt"}},{"binomial",{"logit","probit","cloglog"}},{"gamma",{"inverse_power","log","identity"}}};
            const auto& links=allowed.at(family);
            if(link.empty())link=links.front();
            else if(std::find(links.begin(),links.end(),link)==links.end())bad_param("link "+link+" is not available for family "+family+" (allowed: "+links[0]+", "+links[1]+(links.size()>2?", "+links[2]:"")+(links.size()>3?", "+links[3]:"")+")");
            p["link"]=link;
            if(input.contains("exposure")&&link!="log")bad_param("exposure needs the log link");
            if(input.contains("exposure")&&input.contains("offset"))bad_param("give offset or exposure, not both");
            int_param(input,"maxiter",1,100);double tol=input["tol"];if(!(tol>=1e-12&&tol<=1e-2))bad_param("tol must be in [1e-12, 1e-2]");
        }
    }else if(op=="sm_tsa_tools"){
        sm_name(input.at("series"),"series");const std::string test=input["test"];
        auto given=[&](const char* k){return input.contains(k)&&!input[k].is_null();};
        auto only=[&](std::initializer_list<const char*> keys,const char* where){for(const char* k:keys)if(given(k))bad_param(std::string(k)+" is only used by "+where);};
        if(test=="acf"||test=="pacf"){
            p["alpha"]=given("alpha")?input["alpha"]:Json(0.05);sm_check_alpha(p);
            only({"regression","autolag","maxlag","kpss_lags","lags","model_df"},"adfuller, kpss and ljung_box");
            int64_t nlags=given("nlags")?input["nlags"].get<int64_t>():20;if(nlags<1||nlags>int64_t(SM_ACF_LAGS))bad_param("nlags must be in 1..100");
            p["nlags"]=nlags;p["adjusted"]=given("adjusted")?input["adjusted"]:Json(false);p["method"]=given("method")?input["method"]:Json("ywm");
            if(test=="acf"&&given("method"))bad_param("method is only used by pacf");if(test=="pacf"&&given("adjusted"))bad_param("adjusted is only used by acf");
        }else{
            only({"nlags","adjusted","method","alpha"},"acf and pacf");
            if(test=="adfuller"){
                only({"kpss_lags","lags","model_df"},"kpss and ljung_box");
                p["regression"]=given("regression")?input["regression"]:Json("c");p["autolag"]=given("autolag")?input["autolag"]:Json("AIC");
                if(given("maxlag")){int64_t m=input["maxlag"];if(m<0||m>1000)bad_param("maxlag must be in 0..1000");}
            }else if(test=="kpss"){
                only({"autolag","maxlag","lags","model_df"},"adfuller and ljung_box");
                std::string reg=given("regression")?input["regression"].get<std::string>():"c";if(reg!="c"&&reg!="ct")bad_param("kpss regression is c or ct");p["regression"]=reg;
                Json lags=given("kpss_lags")?input["kpss_lags"]:Json("auto");
                if(lags.is_string()){if(lags!="auto"&&lags!="legacy")bad_param("kpss_lags is auto, legacy or an integer");}
                else if(!lags.is_number_integer()||lags.get<int64_t>()<0||lags.get<int64_t>()>1000)bad_param("kpss_lags is auto, legacy or an integer in 0..1000");
                p["kpss_lags"]=lags;
            }else{
                only({"regression","autolag","maxlag","kpss_lags"},"adfuller and kpss");
                Json lags=given("lags")?input["lags"]:Json(10);int64_t smallest;
                if(lags.is_array()){
                    if(lags.empty()||lags.size()>100)bad_param("lags needs 1..100 entries");smallest=INT64_MAX;
                    int64_t previous=0;for(const auto& v:lags){if(!v.is_number_integer()||v.get<int64_t>()<1||v.get<int64_t>()>100)bad_param("every lag must be an integer in 1..100");if(v.get<int64_t>()<=previous)bad_param("lags must be strictly increasing and unique");previous=v.get<int64_t>();smallest=std::min<int64_t>(smallest,v.get<int64_t>());}
                }else{if(!lags.is_number_integer()||lags.get<int64_t>()<1||lags.get<int64_t>()>100)bad_param("lags must be an integer in 1..100 or a list of them");smallest=lags.get<int64_t>();}
                int64_t df=given("model_df")?input["model_df"].get<int64_t>():0;if(df<0||df>=smallest)bad_param("model_df must be >= 0 and below the smallest lag");
                if(lags.is_number_integer()){int64_t maximum=lags;lags=Json::array();for(int64_t lag=df+1;lag<=maximum;++lag)lags.push_back(lag);}
                p["lags"]=lags;p["model_df"]=df;
            }
        }
    }else if(op=="sm_tsa_acf"){
        sm_name(input.at("series"),"series");int_param(input,"nlags",1,SM_ACF_LAGS);sm_check_alpha(input);
    }else{
        sm_name(input.at("series"),"series");sm_check_alpha(input);
        auto small=[&](const Json& v,int lo,int hi){if(!v.is_number_integer()||v.get<int64_t>()<lo||v.get<int64_t>()>hi)bad_param("ARIMA orders must be integers within the documented caps");return int(v.get<int64_t>());};
        const Json& order=input.at("order");if(!order.is_array()||order.size()!=3)bad_param("order needs [p, d, q]");
        int pp=small(order[0],0,5),d=small(order[1],0,2),q=small(order[2],0,5),P=0,D=0,Q=0,s=0;
        if(input.contains("seasonal_order")&&!input["seasonal_order"].empty()){
            const Json& so=input["seasonal_order"];if(!so.is_array()||so.size()!=4)bad_param("seasonal_order needs [P, D, Q, s]");
            P=small(so[0],0,2);D=small(so[1],0,1);Q=small(so[2],0,2);s=small(so[3],0,24);
            if((P||D||Q)&&s<2)bad_param("seasonal_order needs a period s in 2..24 when P, D or Q is nonzero");
            if(!P&&!D&&!Q&&s!=0&&s<2)bad_param("seasonal period must be 0 or 2..24");
        }
        // SARIMAX state dimension and parameter count, bounded before spawn (the Kalman filter is O(n k^3)).
        const int states=std::max(pp+s*P,q+s*Q+1)+d+s*D,params=pp+q+P+Q+1+(input["trend"]=="n"?0:input["trend"]=="ct"?2:1)+(input.contains("exog")?int(input["exog"].size()):0);
        if(states>int(SM_ARIMA_STATES))invalid("ARIMA state dimension "+std::to_string(states)+" exceeds "+std::to_string(SM_ARIMA_STATES),"INPUT_TOO_LARGE");
        if(params>int(SM_ARIMA_PARAMS))invalid("ARIMA would estimate "+std::to_string(params)+" parameters (limit "+std::to_string(SM_ARIMA_PARAMS)+")","INPUT_TOO_LARGE");
        int_param(input,"maxiter",1,1000);
        int64_t steps=input["forecast_steps"];if(steps<0||steps>int64_t(SM_FORECAST)||(!twin&&steps>int64_t(SM_INLINE_FORECAST)))bad_param(std::string("forecast_steps must be 0..")+(twin?"1000":"100 (use sm_arima_to for up to 1000)"));
        Json names=Json::array();
        if(input.contains("exog")){
            if(!input["exog"].is_array()||input["exog"].size()>SM_EXOG)bad_param("exog needs at most 8 column names");
            std::set<std::string> seen;for(const auto& n:input["exog"]){sm_name(n,"exog");if(!seen.insert(n.get<std::string>()).second)bad_param("duplicate exog column");}
            if(steps>0){
                if(!input.contains("exog_future")||!input["exog_future"].is_object()||input["exog_future"].size()!=input["exog"].size())bad_param("exog_future must give one array per exog column when forecasting");
                for(const auto& n:input["exog"]){
                    const std::string key=n.get<std::string>();
                    if(!input["exog_future"].contains(key)||!input["exog_future"][key].is_array()||int64_t(input["exog_future"][key].size())!=steps)bad_param("exog_future."+key+" must have forecast_steps values");
                    for(const auto& v:input["exog_future"][key])if(!v.is_number()||!std::isfinite(v.get<double>()))bad_param("exog_future values must be finite numbers");
                }
            }
        }else if(input.contains("exog_future"))bad_param("exog_future needs exog");
        if(input.contains("start_params")){
            const Json& sp=input["start_params"];if(!sp.is_array()||sp.empty()||sp.size()>SM_START)bad_param("start_params needs 1..32 finite numbers");
            for(const auto& v:sp)if(!v.is_number()||!std::isfinite(v.get<double>()))bad_param("start_params must be finite numbers");
        }
    }
    if(twin)p["out"]=output_path(input);
    return p;
}
