// S8b front end: la_matrix_fn, la_sparse and special_fn (+ _to twins). Shapes, enums, argument tables, COO indices and every cap are decided here,
// before any process starts; the worker repeats what a corrupted request could break. Reuses the S7/S8 plumbing (array loader, S8Extra,
// tensor write path with the one seven-field artifacts[name] descriptor, time_limit budget).
#pragma once
constexpr size_t S8B_SPARSE_DIM=100000, S8B_SPARSE_NNZ=200000, S8B_SPSOLVE_DIM=20000, S8B_EIG_K=50, S8B_ARTIFACT=1900000;
bool s8b_op(const std::string& op){static const std::set<std::string> ops={"la_matrix_fn","la_matrix_fn_to","la_sparse","la_sparse_to","special_fn","special_fn_to"};return ops.count(op)>0;}
Json prepare_la_matrix_fn(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;Array a=load(in.at("a"));if(a.shape.size()!=2)invalid("a must be a matrix","BAD_SHAPE");need_finite(a,"a");
    const std::string fn=in["fn"];const size_t m=size_t(a.shape[0]),n=size_t(a.shape[1]);
    if(fn!="pinv"&&m!=n)invalid(fn+" needs a square matrix","BAD_SHAPE");
    if(in.contains("rtol")&&!in["rtol"].is_null()){
        if(fn!="pinv")bad_param("rtol is only used by pinv");
        double r=s7_number(in["rtol"],"rtol");if(!(r>=0&&r<1))bad_param("rtol must be in [0, 1)");
    }
    // logm and sqrtm of a real matrix can be complex: two tensors of the same shape
    const size_t numbers=m*n*((fn=="logm"||fn=="sqrtm")?2:1);
    if(!twin)s7_inline_budget(numbers,"la_matrix_fn");
    else if(numbers>S8B_ARTIFACT)output_too_large("the result would hold "+std::to_string(numbers)+" numbers");
    arrays["a"]=std::move(a);return p;
}

struct S8bArg{const char* name;double fallback;bool integer;double lo=0;};  // fallback NAN: required
const std::map<std::string,std::vector<S8bArg>>& s8b_specials(){
    static const std::map<std::string,std::vector<S8bArg>> table={
        {"erf",{{"x",NAN,false}}},{"erfc",{{"x",NAN,false}}},{"erfcx",{{"x",NAN,false}}},{"erfinv",{{"y",NAN,false}}},{"erfcinv",{{"y",NAN,false}}},
        {"gammainc",{{"a",NAN,false},{"x",NAN,false}}},{"gammaincc",{{"a",NAN,false},{"x",NAN,false}}},{"gammaincinv",{{"a",NAN,false},{"y",NAN,false}}},{"gammainccinv",{{"a",NAN,false},{"y",NAN,false}}},
        {"betainc",{{"a",NAN,false},{"b",NAN,false},{"x",NAN,false}}},{"betaincinv",{{"a",NAN,false},{"b",NAN,false},{"y",NAN,false}}},
        {"zeta",{{"x",NAN,false},{"q",1.0,false}}},{"polygamma",{{"n",NAN,true},{"x",NAN,false}}},{"lambertw",{{"z",NAN,false},{"k",0.0,true,-1000.0}}},
        {"expit",{{"x",NAN,false}}},{"logit",{{"p",NAN,false}}},{"log_ndtr",{{"x",NAN,false}}},{"ndtri",{{"p",NAN,false}}},{"ellipk",{{"m",NAN,false}}},{"ellipe",{{"m",NAN,false}}},
        {"hyp2f1",{{"a",NAN,false},{"b",NAN,false},{"c",NAN,false},{"z",NAN,false}}},{"jv",{{"v",NAN,false},{"z",NAN,false}}},{"yv",{{"v",NAN,false},{"z",NAN,false}}},
        {"iv",{{"v",NAN,false},{"z",NAN,false}}},{"kv",{{"v",NAN,false},{"z",NAN,false}}},{"spherical_jn",{{"n",NAN,true},{"z",NAN,false}}},{"airy",{{"x",NAN,false}}},
        {"struve",{{"v",NAN,false},{"x",NAN,false}}},{"owens_t",{{"h",NAN,false},{"a",NAN,false}}},{"boxcox",{{"x",NAN,false},{"lmbda",NAN,false}}},
        {"inv_boxcox",{{"y",NAN,false},{"lmbda",NAN,false}}},{"xlogy",{{"x",NAN,false},{"y",NAN,false}}},{"xlog1py",{{"x",NAN,false},{"y",NAN,false}}},{"logsumexp",{{"a",NAN,false}}}};
    return table;
}
Json prepare_special_fn(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;const std::string fn=in["fn"];const auto& table=s8b_specials();auto spec=table.find(fn);if(spec==table.end())bad_param("unknown function");
    const Json& args=in.at("args");if(!args.is_object())bad_param("args must be an object of argument name -> number or vector");
    std::set<std::string> allowed;for(const auto& a:spec->second)allowed.insert(a.name);
    for(auto it=args.begin();it!=args.end();++it)if(!allowed.count(it.key()))bad_param("argument "+it.key()+" is not used by "+fn);
    Json names=Json::array(),vector=Json::object();size_t length=0;bool any_vector=false;
    for(const auto& a:spec->second){
        Array v;
        if(args.contains(a.name)&&!args[a.name].is_null()){
            const Json& value=args[a.name];
            if(value.is_number()){v.shape={1};v.data={value.get<double>()};}
            else{v=load(value);if(v.shape.size()!=1)invalid(std::string(a.name)+" must be a number or a vector","BAD_DATA");}
        }else if(!std::isnan(a.fallback)){v.shape={1};v.data={a.fallback};}
        else bad_param(fn+" needs argument "+a.name);
        need_finite(v,a.name);
        if(a.integer)for(double d:v.data)if(d!=std::floor(d)||d<a.lo||d>1e9)invalid(std::string(a.name)+" must hold integers >= "+std::to_string(int(a.lo)),"BAD_DATA");
        const bool is_vector=!(args.contains(a.name)&&args[a.name].is_number())&&args.contains(a.name)&&!args[a.name].is_null();
        if(is_vector){if(any_vector&&v.data.size()!=length)invalid("vector arguments must have equal length (scalars broadcast); "+std::string(a.name)+" has "+std::to_string(v.data.size())+" values but another has "+std::to_string(length),"BAD_DATA");length=v.data.size();any_vector=true;}
        names.push_back(a.name);vector[a.name]=is_vector;arrays[std::string("arg_")+a.name]=std::move(v);
    }
    if(fn=="logsumexp"&&!any_vector)bad_param("logsumexp reduces a vector: give a as a vector");
    const size_t n=any_vector?length:1,outputs=fn=="airy"?4*n:fn=="logsumexp"?1:n;
    if(!twin)s7_inline_budget(outputs,"special_fn");else if(outputs>S8B_ARTIFACT)output_too_large("the result would hold "+std::to_string(outputs)+" numbers");
    p["arg_names"]=names;p["arg_vector"]=vector;p.erase("args");return p;
}

Json prepare_la_sparse(const Params& in,std::map<std::string,Array>& arrays,bool twin,S8Extra& x){
    Json p=in;const std::string task=in["task"];
    const bool solve=task=="spsolve"||task=="cg"||task=="gmres",eig=task=="eigs"||task=="eigsh",svd=task=="svds";
    const Json& shape=in.at("shape");
    if(!shape.is_array()||shape.size()!=2||!shape[0].is_number_integer()||!shape[1].is_number_integer())bad_param("shape must be [rows, cols]");
    const int64_t m=shape[0],n=shape[1];
    if(m<1||n<1||size_t(m)>S8B_SPARSE_DIM||size_t(n)>S8B_SPARSE_DIM)invalid("the matrix may have at most 100000 rows and 100000 columns","INPUT_TOO_LARGE");
    if(task!="svds"&&m!=n)invalid(task+" needs a square matrix","BAD_SHAPE");
    if(task=="spsolve"&&size_t(n)>S8B_SPSOLVE_DIM)invalid("spsolve (SuperLU fill-in) is limited to 20000 unknowns; use cg or gmres beyond that","INPUT_TOO_LARGE");
    const bool framed=in.contains("frame")&&!in["frame"].is_null();
    size_t nnz=0;
    if(framed){
        S8Table t=s8_table(in,"frame");
        for(const char* key:{"row","col","val"}){
            if(!in.contains(key)||!in[key].is_string())bad_param(std::string(key)+" must name a column when frame is given");
            std::string c=in[key];s8_need_column(t,c);const std::string d=s8_dtype(t,c);
            if(std::string(key)!="val"&&d!="int64")invalid("column "+c+" must be int64","BAD_DTYPE");
            if(std::string(key)=="val"&&!s8_numeric(d))invalid("column "+c+" must be numeric","BAD_DTYPE");
        }
        nnz=s8_rows(t);if(nnz<1)invalid("the table has no rows","BAD_DATA");
        x.tables["frame"]=std::move(t.bytes);
    }else{
        for(const char* key:{"row","col","val"}){
            if(!in.contains(key)||in[key].is_null())bad_param(std::string(key)+" is required (arrays, or column names with frame)");
            Array a=load(in[key]);if(a.shape.size()!=1)invalid(std::string(key)+" must be a vector","BAD_DATA");need_finite(a,key);
            if(nnz&&a.data.size()!=nnz)invalid("row, col and val must have the same length","BAD_DATA");nnz=a.data.size();arrays[key]=std::move(a);
        }
        const auto& r=arrays.at("row").data;const auto& c=arrays.at("col").data;
        for(size_t i=0;i<nnz;++i){
            if(r[i]!=std::floor(r[i])||c[i]!=std::floor(c[i]))invalid("row and col must hold integer indices","BAD_DATA");
            if(r[i]<0||r[i]>=double(m)||c[i]<0||c[i]>=double(n))invalid("entry "+std::to_string(i)+" lies outside the "+std::to_string(m)+" x "+std::to_string(n)+" matrix","BAD_INDEX");
        }
        if(task=="cg"||task=="eigsh"){
            const auto& v=arrays.at("val").data;std::map<std::pair<int64_t,int64_t>,double> sum;double scale=0;
            for(size_t i=0;i<nnz;++i){sum[{int64_t(r[i]),int64_t(c[i])}]+=v[i];scale=std::max(scale,std::fabs(v[i]));}
            for(const auto& e:sum){auto t=sum.find({e.first.second,e.first.first});double other=t==sum.end()?0.0:t->second;if(std::fabs(e.second-other)>1e-12*scale)invalid(task+" needs a symmetric matrix (entry ("+std::to_string(e.first.first)+", "+std::to_string(e.first.second)+") differs from its transpose)","NOT_SYMMETRIC");}
        }
    }
    if(nnz>S8B_SPARSE_NNZ)invalid("at most 200000 stored entries","INPUT_TOO_LARGE");
    if(solve){
        if(!in.contains("b")||in["b"].is_null())bad_param(task+" needs the right-hand side b");
        Array b=load(in["b"]);if(b.shape.size()!=1||b.data.size()!=size_t(n))invalid("b must be a vector with "+std::to_string(n)+" values","BAD_DATA");need_finite(b,"b");arrays["b"]=std::move(b);
    }else if(in.contains("b")&&!in["b"].is_null())bad_param("b is only used by spsolve, cg and gmres");
    auto given=[&](const char* k){return in.contains(k)&&!in[k].is_null();};
    if(!eig&&!svd){for(const char* k:{"k","which","sigma","ncv","seed"})if(given(k))bad_param(std::string(k)+" is only used by eigs, eigsh and svds");}
    if(task!="gmres"&&given("restart"))bad_param("restart is only used by gmres");
    size_t outputs=n;
    if(eig||svd){
        if(!given("seed"))bad_param(task+" needs an explicit seed (the ARPACK start vector is derived from it with PCG64)");
        if(!in["seed"].is_number_integer()||in["seed"].get<int64_t>()<0)bad_param("seed must be a non-negative integer below 2^63");
        if(!given("k"))bad_param(task+" needs k");
        int64_t k=in["k"];const int64_t limit=task=="eigsh"?n-1:task=="eigs"?n-2:std::min(m,n)-1;
        if(k<1||k>int64_t(S8B_EIG_K)||k>limit)bad_param("k must be 1.."+std::to_string(std::min<int64_t>(S8B_EIG_K,limit))+" for this matrix and task");
        const std::string which=given("which")?in["which"].get<std::string>():"LM";p["which"]=which;
        static const std::map<std::string,std::set<std::string>> allowed={{"eigsh",{"LM","SM","LA","SA","BE"}},{"eigs",{"LM","SM","LR","SR","LI","SI"}},{"svds",{"LM","SM"}}};
        if(!allowed.at(task).count(which))bad_param("which "+which+" is not available for "+task);
        if(given("sigma")){if(svd)bad_param("sigma is not used by svds");if(which!="LM")bad_param("sigma (shift-invert) needs which=LM");s7_number(in["sigma"],"sigma");}
        if(given("ncv")){int64_t ncv=in["ncv"];if(ncv<k+2||ncv>(svd?std::min(m,n):n))bad_param("ncv must be in k+2 .. dimension");}
        const bool vectors=in["vectors"];
        outputs=size_t(k)*(task=="eigs"?2:1);
        if(vectors)outputs+=size_t(k)*(svd?size_t(m+n):size_t(n)*(task=="eigs"?2:1));
    }else if(in["vectors"])bad_param("vectors is only used by eigs, eigsh and svds");
    double rtol;
    if(given("rtol")){rtol=s7_number(in["rtol"],"rtol");if(!(rtol>=0&&rtol<1))bad_param("rtol must be in [0, 1)");if(solve&&rtol==0)bad_param("rtol must be > 0 for cg and gmres");}
    else rtol=solve?1e-8:0.0;
    p["rtol"]=rtol;
    if(given("maxiter")){int64_t it=in["maxiter"];if(it<1||it>1000000)bad_param("maxiter must be 1..1000000");}
    else if(solve)p["maxiter"]=std::min<int64_t>(10*n,100000);
    if(task=="gmres"){int64_t r=given("restart")?in["restart"].get<int64_t>():20;if(r<1||r>1000)bad_param("restart must be 1..1000");p["restart"]=r;}
    p["shape"]=Json::array({m,n});
    if(!twin)s7_inline_budget(outputs,"la_sparse");else if(outputs>S8B_ARTIFACT)output_too_large("the result would hold "+std::to_string(outputs)+" numbers");
    return p;
}
Json prepare_s8b(const std::string& op,const Params& in,std::map<std::string,Array>& arrays,S8Extra& x){
    const bool twin=s7_twin(op);const std::string base=twin?op.substr(0,op.size()-3):op;
    Json p=base=="la_matrix_fn"?prepare_la_matrix_fn(in,arrays,twin):base=="special_fn"?prepare_special_fn(in,arrays,twin):prepare_la_sparse(in,arrays,twin,x);
    if(twin){p["out"]=output_path(in);x.tensor_write=true;}
    return p;
}

// ---- registration ----
void add_s8b_ops(CliSpec& spec){
    auto str=[](const char* name,bool required,Json def,std::vector<Json> values,const char* text){return Param{name,Type::String,required,std::move(def),std::move(values),text};};
    auto json=[](const char* name,bool required,const char* text){return Param{name,Type::Json,required,nullptr,{},text};};
    auto num=[](const char* name,const char* text){return Param{name,Type::Number,false,nullptr,{},text};};
    auto integer=[](const char* name,const char* text){return Param{name,Type::Integer,false,nullptr,{},text};};
    const char* out_text="New .safetensors path inside the working directory (CONFLICT if it exists). The write form stores the arrays there and returns the artifacts[name] descriptors.";
    for(bool write:{false,true}){
        auto finish=[&](Operation op){if(write)op.params.emplace_back("out",Type::String,true,nullptr,std::vector<Json>{},out_text);spec.operations.push_back(std::move(op));};
        const char* tail=write?" Writes the arrays to an exclusive safetensors artifact (nothing is written for a non-OK status).":" Inline results hold at most ~900 numbers; use the _to form for more.";
        finish(Operation{write?"la_matrix_fn_to":"la_matrix_fn",write?Access::Write:Access::Read,std::string("Matrix functions of a finite real matrix (scipy.linalg): inv, pinv (rtol cutoff, reports rank), expm, logm and sqrtm (principal values; a result that is not exactly real is split into y_re / y_im and complex=true). Verified after the fact: inv (condition number; singular to working precision is SINGULAR), pinv (A Y A = A), logm (expm(Y) = A) and sqrtm (Y Y = A) report the relative residual and fail with NUMERICAL_FAILURE above 1e-6; non-finite results and logm of a singular matrix are statuses. Statuses OK, SINGULAR, NUMERICAL_FAILURE, TIME_LIMIT. There is no funm: an arbitrary function would need a callable.")+tail,"object",{
            json("a",true,"Finite matrix (inline <=1000 numbers, .npy or safetensors#key, <=200000 numbers); square for every function except pinv."),
            str("fn",true,nullptr,{"inv","pinv","expm","logm","sqrtm"},"Matrix function."),
            num("rtol","pinv only: singular values below rtol * the largest are treated as zero, in [0, 1) (default max(m, n) * machine epsilon).")}});
        finish(Operation{write?"special_fn_to":"special_fn",write?Access::Write:Access::Read,std::string("Special functions from a fixed allowlist (scipy.special) on scalar or vector arguments. Broadcasting rule: every vector argument has the same length n and scalars repeat; the result has n values (airy: four vectors ai, aip, bi, bip; logsumexp reduces its vector to one value). Defaults: zeta q=1 (Riemann), lambertw branch k=0 (a result with a non-zero imaginary part is a domain error). Any result that is NaN (outside the function's domain) makes the status DOMAIN_ERROR, and any infinite result (pole or overflow) NON_FINITE, with the count, the first index and its arguments: special-function singularities are never returned as values. Statuses OK, DOMAIN_ERROR, NON_FINITE, TIME_LIMIT.")+tail,"object",{
            str("fn",true,nullptr,{"erf","erfc","erfcx","erfinv","erfcinv","gammainc","gammaincc","gammaincinv","gammainccinv","betainc","betaincinv","zeta","polygamma","lambertw","expit","logit","log_ndtr","ndtri","ellipk","ellipe","hyp2f1","jv","yv","iv","kv","spherical_jn","airy","struve","owens_t","boxcox","inv_boxcox","xlogy","xlog1py","logsumexp"},"Function. Arguments by name: erf, erfc, erfcx, expit, log_ndtr, airy: x; erfinv, erfcinv: y; logit, ndtri: p; gammainc, gammaincc: a, x; gammaincinv, gammainccinv: a, y; betainc: a, b, x; betaincinv: a, b, y; zeta: x, q; polygamma: n (integer >= 0), x; lambertw: z, k; ellipk, ellipe: m; hyp2f1: a, b, c, z; jv, yv, iv, kv: v, z; spherical_jn: n (integer >= 0), z; struve: v, x; owens_t: h, a; boxcox: x, lmbda; inv_boxcox: y, lmbda; xlogy, xlog1py: x, y; logsumexp: a (a vector)."),
            Param{"args",Type::Object,true,nullptr,{},"Argument name -> number or vector (inline <=1000 numbers, .npy or safetensors#key, <=200000); finite values only. Unknown or missing required names are BAD_PARAM; vector lengths must agree (BAD_DATA)."}}});
        finish(Operation{write?"la_sparse_to":"la_sparse",write?Access::Write:Access::Read,std::string("Sparse linear algebra on a COO matrix (scipy.sparse; duplicate entries are summed): spsolve (SuperLU, n <= 20000), cg (symmetric, checked), gmres, and ARPACK eigsh (symmetric), eigs and svds with a start vector derived from the required seed via PCG64 (reproducible). The true residual is recomputed: cg/gmres report relative_residual and are NO_CONVERGENCE when it misses the tolerance, spsolve is SINGULAR for an exactly singular matrix and ILL_CONDITIONED above 1e-6, eigenpairs report max_residual. ARPACK non-convergence is a domain status (n_converged reported, no partial values). eigs results are sorted by (real, imaginary), eigsh ascending, svds descending. Statuses OK, NO_CONVERGENCE, SINGULAR, ILL_CONDITIONED, NUMERICAL_FAILURE, TIME_LIMIT. Result arrays: x; w (w_re / w_im when complex) and v; s, u, vt.")+tail,"object",{
            str("task",true,nullptr,{"spsolve","cg","gmres","eigsh","eigs","svds"},"Operation."),
            json("shape",true,"[rows, cols], each 1..100000 (square except for svds)."),
            json("row",true,"Row indices of the entries: an integer vector (inline, .npy, safetensors#key; <=200000 entries, in range) or, with frame, the int64 column name."),
            json("col",true,"Column indices, like row."),
            json("val",true,"Finite entry values, like row (a numeric column name with frame)."),
            str("frame",false,nullptr,{},"Optional .npz table holding the entries; row, col and val then name its columns."),
            json("b",false,"Right-hand side vector of length n (spsolve, cg, gmres only)."),
            integer("k","eigs, eigsh, svds: number of eigenvalues / singular values, 1..50 and below the dimension (eigsh < n, eigs < n-1, svds < min(rows, cols))."),
            str("which",false,nullptr,{"LM","SM","LA","SA","BE","LR","SR","LI","SI"},"Which values ARPACK returns (default LM): eigsh LM, SM, LA, SA, BE; eigs LM, SM, LR, SR, LI, SI; svds LM, SM."),
            num("sigma","eigs / eigsh: shift for shift-invert mode (needs which=LM); eigenvalues nearest sigma are returned."),
            integer("ncv","eigs, eigsh, svds: Lanczos basis size, k+2 .. dimension (default ARPACK's)."),
            integer("seed","eigs, eigsh, svds: REQUIRED non-negative integer; the ARPACK start vector is numpy PCG64(seed).uniform(-1, 1), so equal seeds give equal results."),
            num("rtol","cg / gmres: relative residual tolerance in (0, 1), default 1e-8; ARPACK: tolerance in [0, 1), default 0 (machine precision)."),
            integer("maxiter","Iteration cap, 1..1000000 (cg / gmres default min(10 n, 100000); ARPACK default 10 n)."),
            integer("restart","gmres: restart length, 1..1000 (default 20)."),
            Param{"vectors",Type::Boolean,false,false,{},"eigs, eigsh, svds: also return the vectors v (n x k), or u (rows x k) and vt (k x cols)."}}});
    }
}
