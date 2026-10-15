// S7 front end: interpolation, signal processing and spatial operations. Every array is loaded, shape-checked and finite-checked here;
// every enum, QHull option string, order, window and cap is decided here, so the worker never sees a request that could exhaust memory.
#pragma once
constexpr size_t S7_INLINE=900, S7_RBF_POINTS=5000, S7_RBF_DIM=8, S7_KNN_DIM=32, S7_ORDER=20, S7_NPERSEG=65536, S7_NFFT=131072, S7_DIST_PAIRS=1000000, S7_FREQS=200, S7_KNN_K=1000, S7_HULL_DIM=6, S7_ARTIFACT=1900000, S7_BRUTE_WORK=20000000;
const std::vector<std::string> S7_WINDOWS={"hann","hamming","blackman","boxcar","bartlett","flattop","parzen","bohman","blackmanharris","nuttall","barthann","cosine","triang"};
bool s7_twin(const std::string& op){return op.size()>3&&op.compare(op.size()-3,3,"_to")==0;}
bool s7_op(const std::string& op){return op.rfind("interp_",0)==0||op.rfind("sig_",0)==0||op.rfind("spatial_",0)==0;}
const Array& s7_load(const Params& in,const char* key,std::map<std::string,Array>& arrays,size_t rank,size_t min_items=1){
    Array a=load(in.at(key));
    if(rank==2&&a.shape.size()==1)a.shape={int64_t(a.data.size()),1};  // a vector is a one-column point set
    if(a.shape.size()!=rank)invalid(std::string(key)+(rank==1?" must be a vector":" must be a matrix (rows are points)"),"BAD_DATA");
    need_finite(a,key);
    if(a.data.size()<min_items)invalid(std::string(key)+" has too few values","BAD_DATA");
    arrays[key]=std::move(a);return arrays[key];
}
double s7_number(const Json& v,const std::string& name){if(!v.is_number()||!std::isfinite(v.get<double>()))bad_param(name+" must be a finite number");return v.get<double>();}
// number | [lower|null, upper|null]: returned as a JSON pair with nulls kept; low<=high; `non_negative` for threshold-like properties
Json s7_range(const Json& v,const std::string& name,bool non_negative){
    auto check=[&](const Json& x){if(x.is_null())return;double d=s7_number(x,name);if(non_negative&&d<0)bad_param(name+" must be >= 0");};
    if(v.is_array()){
        if(v.size()!=2)bad_param(name+" needs a number or [lower, upper] (null for an open end)");
        check(v[0]);check(v[1]);if(!v[0].is_null()&&!v[1].is_null()&&v[0].get<double>()>v[1].get<double>())bad_param(name+" lower bound exceeds upper bound");
        return v;
    }
    check(v);if(v.is_null())bad_param(name+" must be a number or a pair");return v;
}
void s7_reject_unused(const Params& in,std::initializer_list<const char*> keys,const std::string& why){for(const char* key:keys)if(in.contains(key)&&!in[key].is_null())bad_param(std::string(key)+" is only used "+why);}
void s7_inline_budget(size_t numbers,const std::string& op){if(numbers>S7_INLINE)output_too_large("the "+op+" result would hold about "+std::to_string(numbers)+" inline numbers (limit 1000); use "+op+"_to");}

Json prepare_interp(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;const std::string kind=in["kind"];const bool rbf=kind=="rbf";
    const Array& y=s7_load(in,"y",arrays,1);const size_t n=y.data.size();
    const Array& x=s7_load(in,"x",arrays,rbf?2:1);
    if(rbf){
        if(size_t(x.shape[0])!=n)invalid("x needs one point (row) per y value","BAD_DATA");
        if(n>S7_RBF_POINTS||size_t(x.shape[1])>S7_RBF_DIM)invalid("rbf allows at most "+std::to_string(S7_RBF_POINTS)+" points in at most "+std::to_string(S7_RBF_DIM)+" dimensions (the solve is O(n^3))","INPUT_TOO_LARGE");
    }else{
        const size_t minimum=kind=="smoothing"?5:2;
        if(x.data.size()!=n)invalid("x and y must have the same length","BAD_DATA");
        if(n<minimum)invalid(kind+" interpolation needs at least "+std::to_string(minimum)+" points","BAD_DATA");
        for(size_t i=1;i<n;++i){
            if(x.data[i]==x.data[i-1])invalid("x contains a duplicate value at index "+std::to_string(i)+": nodes must be strictly increasing","DUPLICATE_X");
            if(x.data[i]<x.data[i-1])invalid("x is not increasing at index "+std::to_string(i)+": sort the nodes first","NOT_INCREASING");
        }
    }
    const int64_t nu=in["nu"];const int64_t max_nu=rbf?0:kind=="linear"?1:3;
    if(nu<0||nu>max_nu)bad_param("nu must be 0.."+std::to_string(max_nu)+" for kind "+kind);
    // Options belong to exactly one kind; a stray option is an error, never silently ignored.
    if(kind=="cubic"){
        std::string bc=in.contains("bc_type")&&!in["bc_type"].is_null()?in["bc_type"].get<std::string>():"not-a-knot";p["bc_type"]=bc;
        if(bc=="periodic"&&y.data.front()!=y.data.back())invalid("periodic boundary conditions need y[0] == y[-1]","BAD_DATA");
    }else if(in.contains("bc_type")&&!in["bc_type"].is_null())bad_param("bc_type is only used with kind cubic");
    if(in.contains("lam")&&!in["lam"].is_null()){if(kind!="smoothing")bad_param("lam is only used with kind smoothing");double lam=s7_number(in["lam"],"lam");if(lam<0)bad_param("lam must be >= 0");
        // scipy's banded solver silently loses all accuracy for a huge penalty (measured: error 0.18 at lam/h^3 = 1.6e13 on unit-scale data), so the
        // penalty relative to the mean node spacing is capped; at the cap the fit is already the least-squares line to ~1e-5.
        const double h=(x.data.back()-x.data.front())/double(n-1);
        if(lam/(h*h*h)>1e11)bad_param("lam is too large for these nodes (lam / spacing^3 must be <= 1e11, i.e. lam <= "+std::to_string(1e11*h*h*h)+"): the spline is numerically the least-squares line");}
    if(rbf){
        std::string kernel=in.contains("kernel")&&!in["kernel"].is_null()?in["kernel"].get<std::string>():"thin_plate_spline";p["kernel"]=kernel;
        double epsilon=in.contains("epsilon")&&!in["epsilon"].is_null()?s7_number(in["epsilon"],"epsilon"):1.0;if(!(epsilon>0))bad_param("epsilon must be > 0");p["epsilon"]=epsilon;
        double smoothing=in.contains("smoothing")&&!in["smoothing"].is_null()?s7_number(in["smoothing"],"smoothing"):0.0;if(smoothing<0)bad_param("smoothing must be >= 0");p["smoothing"]=smoothing;
        if(in.contains("degree")&&!in["degree"].is_null()){int64_t d=in["degree"];if(d<-1||d>3)bad_param("degree must be -1..3");}
        int minimum=kernel=="quintic"?2:(kernel=="thin_plate_spline"||kernel=="cubic")?1:0;
        int degree=in.contains("degree")&&!in["degree"].is_null()?in["degree"].get<int>():minimum;
        if(degree!=-1&&degree<minimum)bad_param("rbf degree is below this kernel's polynomial minimum");
        size_t terms=1;for(int j=1;j<=degree;++j)terms=terms*(size_t(x.shape[1])+j)/j;
        if(degree>=0&&n<terms)bad_param("too few rbf nodes for polynomial degree: need at least "+std::to_string(terms));
        p["degree"]=degree;
        if(in.contains("integrate")&&!in["integrate"].is_null())bad_param("integrate is not available for kind rbf");
    }else{
        for(const char* key:{"kernel","epsilon","smoothing","degree"})if(in.contains(key)&&!in[key].is_null())bad_param(std::string(key)+" is only used with kind rbf");
    }
    const std::string extrapolate=in["extrapolate"];
    const bool has_new=in.contains("x_new")&&!in["x_new"].is_null(),has_int=in.contains("integrate")&&!in["integrate"].is_null();
    if(!has_new&&!has_int)bad_param("give x_new, integrate or both");
    if(twin&&!has_new)bad_param("interp_eval_to needs x_new; integrate-only requests use interp_eval");
    size_t outputs=0;
    if(has_new){
        const Array& q=s7_load(in,"x_new",arrays,rbf?2:1);
        if(rbf&&q.shape[1]!=x.shape[1])invalid("x_new points need "+std::to_string(x.shape[1])+" coordinates","BAD_DATA");
        const size_t m=rbf?size_t(q.shape[0]):q.data.size();outputs+=m;
        if(extrapolate=="error"){
            if(rbf){
                const size_t d=size_t(x.shape[1]);
                for(size_t j=0;j<d;++j){double lo=INFINITY,hi=-INFINITY;for(size_t i=0;i<n;++i){lo=std::min(lo,x.data[i*d+j]);hi=std::max(hi,x.data[i*d+j]);}
                    for(size_t i=0;i<m;++i){double v=q.data[i*d+j];if(v<lo||v>hi)invalid("x_new row "+std::to_string(i)+" lies outside the bounding box of the nodes; set extrapolate to nan or extend","BAD_RANGE");}}
            }else for(size_t i=0;i<m;++i)if(q.data[i]<x.data.front()||q.data[i]>x.data.back())invalid("x_new["+std::to_string(i)+"] lies outside ["+std::to_string(x.data.front())+", "+std::to_string(x.data.back())+"]; set extrapolate to nan or extend","BAD_RANGE");
        }
    }
    if(has_int){
        const Json& b=in["integrate"];
        if(!b.is_array()||b.size()!=2)bad_param("integrate needs [a, b]");
        double a=s7_number(b[0],"integrate[0]"),c=s7_number(b[1],"integrate[1]");if(!(a<c))bad_param("integrate needs a < b");
        if(extrapolate!="extend"&&(a<x.data.front()||c>x.data.back()))invalid("integrate bounds lie outside the node range; set extrapolate to extend","BAD_RANGE");
        outputs+=1;
    }
    if(!twin)s7_inline_budget(outputs,"interp_eval");
    return p;
}

Json prepare_filter(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;const std::string family=in["family"],btype=in["btype"];const int64_t order=in["order"];
    if(order<1||order>int64_t(S7_ORDER))bad_param("order must be 1..20");
    const bool band=btype=="bandpass"||btype=="bandstop";
    double nyquist=1.0;
    if(in.contains("fs")&&!in["fs"].is_null()){double fs=s7_number(in["fs"],"fs");if(!(fs>0))bad_param("fs must be > 0");nyquist=fs/2;}
    const Json& wn=in.at("wn");std::vector<double> edges;
    if(wn.is_array()){for(const auto& v:wn)edges.push_back(s7_number(v,"wn"));}else edges.push_back(s7_number(wn,"wn"));
    if(band!=(edges.size()==2)||edges.size()>2||edges.empty())bad_param(band?"band filters need wn = [low, high]":"low/high-pass filters need a single wn");
    for(double e:edges)if(!(e>0&&e<nyquist))bad_param("wn must lie strictly between 0 and the Nyquist frequency ("+std::to_string(nyquist)+")");
    if(band&&!(edges[0]<edges[1]))bad_param("wn must be increasing: [low, high]");
    const bool needs_rp=family=="cheby1"||family=="ellip",needs_rs=family=="cheby2"||family=="ellip";
    auto ripple=[&](const char* key,bool needed,double hi){
        const bool given=in.contains(key)&&!in[key].is_null();
        if(needed&&!given)bad_param(std::string(key)+" is required for family "+family);
        if(!needed&&given)bad_param(std::string(key)+" is not used by family "+family);
        if(given){double v=s7_number(in[key],key);if(!(v>0&&v<=hi))bad_param(std::string(key)+" must be in (0, "+std::to_string(int(hi))+"] dB");}
    };
    ripple("rp",needs_rp,100);ripple("rs",needs_rs,200);
    if(family=="ellip"&&in["rp"].get<double>()>=in["rs"].get<double>())bad_param("ellip requires rp < rs");
    const size_t sections=band?size_t(order):size_t((order+1)/2);size_t numbers=6*sections;
    const bool has_x=in.contains("x")&&!in["x"].is_null();
    if(in.contains("mode")&&!in["mode"].is_null()&&!has_x)bad_param("mode is only used with x");
    if(has_x){
        const Array& x=s7_load(in,"x",arrays,1);const std::string mode=in.contains("mode")&&!in["mode"].is_null()?in["mode"].get<std::string>():"sosfiltfilt";p["mode"]=mode;
        if(mode=="sosfiltfilt"&&x.data.size()<=3*(2*sections+1))invalid("sosfiltfilt needs more than "+std::to_string(3*(2*sections+1))+" samples for this filter (edge padding)","BAD_DATA");
        numbers+=x.data.size();
    }else if(twin)bad_param("the write form needs x");
    if(in.contains("freqs")&&!in["freqs"].is_null()){
        Array f=load(in["freqs"]);if(f.shape.size()!=1)invalid("freqs must be a vector","BAD_DATA");if(f.data.size()>S7_FREQS)invalid("freqs exceeds 200 values","INPUT_TOO_LARGE");need_finite(f,"freqs");
        for(double v:f.data)if(v<0||v>nyquist)bad_param("freqs must lie in [0, Nyquist]");
        numbers+=2*f.data.size();arrays["freqs"]=std::move(f);
    }
    if(!twin)s7_inline_budget(numbers,"sig_filter");
    return p;
}

Json prepare_spectrum(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;const std::string method=in["method"];const bool periodogram=method=="periodogram",needs_y=method=="csd"||method=="coherence";
    const Array& x=s7_load(in,"x",arrays,1,4);const size_t n=x.data.size();
    if(!(in["fs"].get<double>()>0&&std::isfinite(in["fs"].get<double>())))bad_param("fs must be > 0");
    const bool has_y=in.contains("y")&&!in["y"].is_null();
    if(needs_y&&!has_y)bad_param("method "+method+" needs y");
    if(!needs_y&&has_y)bad_param("y is only used with csd and coherence");
    if(has_y){const Array& y=s7_load(in,"y",arrays,1);if(y.data.size()!=n)invalid("x and y must have the same length","BAD_DATA");}
    if(in.contains("window")&&!in["window"].is_null()){const std::string w=in["window"];if(std::find(S7_WINDOWS.begin(),S7_WINDOWS.end(),w)==S7_WINDOWS.end())bad_param("unknown window");}
    const bool has_scaling=in.contains("scaling")&&!in["scaling"].is_null();
    if(has_scaling&&method=="coherence")bad_param("scaling is not used by coherence");
    p["scaling"]=has_scaling?in["scaling"]:Json("density");
    const bool has_average=in.contains("average")&&!in["average"].is_null();
    if(has_average&&(periodogram||method=="coherence"))bad_param("average is only used with welch and csd");
    p["average"]=has_average?in["average"]:Json("mean");
    size_t effective;
    if(periodogram){
        s7_reject_unused(in,{"nperseg","noverlap"},"with welch, csd and coherence");
        effective=n;
    }else{
        size_t nperseg=in.contains("nperseg")&&!in["nperseg"].is_null()?size_t(int_param(in,"nperseg",2,1000000000)):std::min<size_t>(256,method=="coherence"?n/2:n);
        if(nperseg>S7_NPERSEG)invalid("nperseg is limited to 65536","INPUT_TOO_LARGE");
        if(nperseg>n)bad_param("nperseg exceeds the signal length");
        size_t noverlap=in.contains("noverlap")&&!in["noverlap"].is_null()?size_t(int_param(in,"noverlap",0,1000000000)):nperseg/2;
        if(noverlap>=nperseg)bad_param("noverlap must be smaller than nperseg");
        if(method=="coherence"&&1+(n-nperseg)/(nperseg-noverlap)<2)bad_param("coherence requires at least two averaged segments (nperseg < signal length)");
        p["nperseg"]=nperseg;p["noverlap"]=noverlap;effective=nperseg;
    }
    if(in.contains("nfft")&&!in["nfft"].is_null()){
        int64_t nfft=in["nfft"];if(nfft<1)bad_param("nfft must be >= 1");if(size_t(nfft)>S7_NFFT)invalid("nfft is limited to 131072","INPUT_TOO_LARGE");
        if(size_t(nfft)<effective)bad_param("nfft must be at least "+std::string(periodogram?"the signal length":"nperseg"));
        effective=size_t(nfft);
    }
    const size_t bins=effective/2+1;
    if(!twin)s7_inline_budget((method=="csd"?3:2)*bins,"sig_spectrum");
    return p;
}

Json prepare_peaks(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;s7_load(in,"x",arrays,1,3);
    for(const char* key:{"height","threshold","prominence","width"})if(in.contains(key)&&!in[key].is_null())p[key]=s7_range(in[key],key,std::string(key)!="height");
    if(in.contains("distance")&&!in["distance"].is_null()){double d=s7_number(in["distance"],"distance");if(d<1)bad_param("distance must be >= 1 sample");}
    if(in.contains("wlen")&&!in["wlen"].is_null()){int64_t w=in["wlen"];if(w<3)bad_param("wlen must be >= 3 samples");}
    double rel=in["rel_height"];if(!(rel>0&&rel<=1))bad_param("rel_height must be in (0, 1]");
    return p;
}

Json prepare_knn(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;const Array& pts=s7_load(in,"points",arrays,2);const size_t n=size_t(pts.shape[0]),d=size_t(pts.shape[1]);
    if(d>S7_KNN_DIM)invalid("points allow at most 32 dimensions","INPUT_TOO_LARGE");
    const bool has_q=in.contains("queries")&&!in["queries"].is_null();size_t m=n;
    if(has_q){const Array& q=s7_load(in,"queries",arrays,2);if(size_t(q.shape[1])!=d)invalid("queries need "+std::to_string(d)+" coordinates per row","BAD_DATA");m=size_t(q.shape[0]);}
    const bool has_k=in.contains("k")&&!in["k"].is_null(),has_r=in.contains("radius")&&!in["radius"].is_null();
    if(has_k==has_r)bad_param("give exactly one of k and radius");
    const bool exclude=in["exclude_self"];if(exclude&&has_q)bad_param("exclude_self is only used when queries are omitted (the points query themselves)");
    const Json& norm=in.at("p");
    if(norm.is_string()){if(norm!="inf")bad_param("p must be a number in [1, 64] or \"inf\"");}
    else{double v=s7_number(norm,"p");if(v<1||v>64)bad_param("p must be a number in [1, 64] or \"inf\"");}
    if(has_k){
        int64_t k=in["k"];if(k<1||size_t(k)>S7_KNN_K)bad_param("k must be 1..1000");
        if(size_t(k)+(exclude?1:0)>n)bad_param("k"+std::string(exclude?" + 1":"")+" exceeds the number of points ("+std::to_string(n)+")");
        if(2*m*(size_t(k)+(exclude?1:0))>S7_ARTIFACT)output_too_large("spatial_knn query workspace/output exceeds 1900000 numbers; reduce queries or k");
        if(!twin)s7_inline_budget(2*m*size_t(k),"spatial_knn");
    }else{double r=s7_number(in["radius"],"radius");if(!(r>0))bad_param("radius must be > 0");if(!twin)s7_inline_budget(m,"spatial_knn");}
    if(!norm.is_string()&&norm.get<double>()>2&&n*m*d>S7_BRUTE_WORK)invalid("p > 2 uses stable exact distances: n*m*d exceeds 20000000","INPUT_TOO_LARGE");
    int_param(in,"max_pairs",1,1000000);
    return p;
}

Json prepare_hull(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;const Array& pts=s7_load(in,"points",arrays,2);const size_t n=size_t(pts.shape[0]),d=size_t(pts.shape[1]);
    if(d<2)invalid("points need at least 2 dimensions","BAD_DATA");
    if(d>S7_HULL_DIM)invalid("points allow at most 6 dimensions","INPUT_TOO_LARGE");
    const std::string kind=in["kind"];
    // Output size explodes with dimension (a cyclic polytope has O(n^(d/2)) facets; measured worst cases in docs/timings.md), so the caps shrink with d
    // until the worst-case simplex list still fits the 2M-number artifact budget.
    static const size_t hull_cap[7]={0,0,20000,20000,800,400,100},triangulation_cap[7]={0,0,20000,800,300,100,50};
    const size_t cap=(kind=="convex_hull"?hull_cap:triangulation_cap)[d];
    if(n>cap)invalid(kind+" in "+std::to_string(d)+" dimensions allows at most "+std::to_string(cap)+" points","INPUT_TOO_LARGE");
    // The only QHull options that ever reach the library: a fixed table per kind (and Qx for d > 4, scipy's own rule). Never caller-supplied.
    const std::string extra=d>4?" Qx":"";
    p["qhull_options"]=kind=="convex_hull"?"Qt"+extra:kind=="delaunay"?"Qbb Qc Qz Q12 Qt"+extra:"Qbb Qc Qz Q12"+extra;
    return p;
}

Json prepare_dist(const Params& in,std::map<std::string,Array>& arrays,bool twin){
    Json p=in;const std::string metric=in["metric"];const Array& a=s7_load(in,"a",arrays,2);const size_t d=size_t(a.shape[1]);size_t m=size_t(a.shape[0]);
    if(in.contains("b")&&!in["b"].is_null()){const Array& b=s7_load(in,"b",arrays,2);if(size_t(b.shape[1])!=d)invalid("a and b need the same number of columns","BAD_DATA");m=size_t(b.shape[0]);}
    const bool has_vi=in.contains("vi")&&!in["vi"].is_null();
    if(metric=="mahalanobis"){
        if(!has_vi)bad_param("metric mahalanobis needs vi (the inverse covariance matrix)");
        const Array& vi=s7_load(in,"vi",arrays,2);if(size_t(vi.shape[0])!=d||size_t(vi.shape[1])!=d)invalid("vi must be "+std::to_string(d)+" x "+std::to_string(d),"BAD_DATA");
        double scale=0;for(double v:vi.data)scale=std::max(scale,std::abs(v));bool changed=false;
        for(size_t i=0;i<d;++i)for(size_t j=0;j<i;++j){double a=vi.data[i*d+j],b=vi.data[j*d+i];if(std::abs(a-b)>1e-12*scale)invalid("vi must be symmetric to relative tolerance 1e-12","NOT_SYMMETRIC");if(a!=b){changed=true;arrays["vi"].data[i*d+j]=arrays["vi"].data[j*d+i]=a/2+b/2;}}
        p["vi_symmetrised"]=changed;
    }else if(has_vi)bad_param("vi is only used with metric mahalanobis");
    const size_t pairs=size_t(a.shape[0])*m;
    if(pairs>S7_DIST_PAIRS)invalid("the distance matrix would hold "+std::to_string(pairs)+" entries (limit "+std::to_string(S7_DIST_PAIRS)+")","INPUT_TOO_LARGE");
    if(!twin)s7_inline_budget(pairs,"spatial_dist");
    return p;
}

Json prepare_s7(const std::string& op,const Params& in,std::map<std::string,Array>& arrays){
    const bool twin=s7_twin(op);const std::string base=twin?op.substr(0,op.size()-3):op;
    Json p=base=="interp_eval"?prepare_interp(in,arrays,twin):base=="sig_filter"?prepare_filter(in,arrays,twin):base=="sig_spectrum"?prepare_spectrum(in,arrays,twin):base=="sig_peaks"?prepare_peaks(in,arrays,twin)
        :base=="spatial_knn"?prepare_knn(in,arrays,twin):base=="spatial_hull"?prepare_hull(in,arrays,twin):prepare_dist(in,arrays,twin);
    if(twin)p["out"]=output_path(in);
    return p;
}

// ---- registration ----
void add_s7_ops(CliSpec& spec){
    auto num=[](const char* name,const char* text){return Param{name,Type::Number,false,nullptr,{},text};};
    auto integer=[](const char* name,Json def,const char* text){return Param{name,Type::Integer,false,std::move(def),{},text};};
    auto json=[](const char* name,bool required,const char* text){return Param{name,Type::Json,required,nullptr,{},text};};
    auto text=[](const char* name,bool required,Json def,std::vector<Json> values,const char* description){return Param{name,Type::String,required,std::move(def),std::move(values),description};};
    const char* out_text="New .safetensors path inside the working directory (CONFLICT if it exists). The write form stores the large arrays there and returns descriptors under tensors.";
    for(bool write:{false,true}){
        auto finish=[&](Operation op){if(write)op.params.emplace_back("out",Type::String,true,nullptr,std::vector<Json>{},out_text);spec.operations.push_back(std::move(op));};
        const char* tail=write?" Writes the arrays to an exclusive safetensors artifact (nothing is written for a non-OK status).":" Inline results hold at most ~900 numbers; use the _to form for more.";
        finish(Operation{write?"interp_eval_to":"interp_eval",write?Access::Write:Access::Read,std::string("Evaluate, differentiate or integrate an interpolant (scipy.interpolate): linear, cubic spline, PCHIP (monotone), Akima, modified Akima, smoothing spline or n-D radial basis functions. Nodes must be finite and strictly increasing (NOT_INCREASING / DUPLICATE_X before any work); outside the node range the extrapolate policy decides (BAD_RANGE by default). Statuses OK, SINGULAR / ILL_CONDITIONED (rbf), NUMERICAL_FAILURE, TIME_LIMIT. Result: values, n_extrapolated, integral, interpolation_error (rbf).")+tail,"object",{
            text("kind",true,nullptr,{"linear","cubic","pchip","akima","makima","smoothing","rbf"},"Interpolant. linear/cubic/pchip/akima/makima need >= 2 nodes, smoothing >= 5, rbf at most 5000 nodes in at most 8 dimensions. pchip/akima/makima do not overshoot step-like data; cubic does."),
            json("x",true,"Nodes: a strictly increasing vector (kinds other than rbf) or, for rbf, a matrix with one row per node. Finite; inline <=1000 numbers or .npy / .safetensors#key."),
            json("y",true,"Finite values at the nodes: a vector with one value per node."),
            json("x_new",false,"Evaluation points: a vector (a matrix with the node dimension for rbf). Give x_new, integrate or both. Read form: at most ~900 points."),
            text("bc_type",false,nullptr,{"not-a-knot","natural","clamped","periodic"},"cubic only. Boundary condition (default not-a-knot): natural sets the second derivative to 0 at the ends, clamped the first derivative to 0, periodic needs y[0] == y[-1]."),
            num("lam","smoothing only. Penalty weight >= 0; 0 interpolates (natural cubic spline), a large value tends to the least-squares line. Omitted: chosen by generalised cross-validation."),
            text("kernel",false,nullptr,{"linear","thin_plate_spline","cubic","quintic","multiquadric","inverse_multiquadric","inverse_quadratic","gaussian"},"rbf only. Radial basis function (default thin_plate_spline). thin_plate_spline, cubic and quintic reproduce low-degree polynomials exactly."),
            num("epsilon","rbf only. Shape parameter > 0 (default 1) used by multiquadric, inverse_multiquadric, inverse_quadratic and gaussian."),
            num("smoothing","rbf only. Ridge term >= 0 (default 0 = interpolate the nodes)."),
            integer("degree",nullptr,"rbf only. Degree of the polynomial term, -1..3 (default: the kernel's minimum)."),
            integer("nu",0,"Derivative order: 0..3 (linear: 0..1; rbf: 0). At knots a piecewise function takes its right-hand value."),
            json("integrate",false,"[a, b] with a < b: the definite integral of the interpolant (not for rbf). Bounds must lie in the node range unless extrapolate is extend."),
            text("extrapolate",false,"error",{"error","nan","extend"},"Points outside the nodes: error (BAD_RANGE, the default), nan (NaN values, counted in n_extrapolated) or extend (the end polynomial; for rbf the bounding box of the nodes is the 'range').")}});
        finish(Operation{write?"sig_filter_to":"sig_filter",write?Access::Write:Access::Read,std::string("Design an IIR filter as second-order sections (scipy.signal.iirfilter, output sos; never ba) and optionally apply it. Reports sos, max_pole_radius and stable; an unstable design returns status UNSTABLE and is not applied. Optional magnitude/phase at freqs. Statuses OK, UNSTABLE, NUMERICAL_FAILURE.")+tail,"object",{
            text("family",true,nullptr,{"butter","cheby1","cheby2","ellip","bessel"},"Prototype. cheby1 needs rp, cheby2 rs, ellip both."),
            Param{"order",Type::Integer,true,nullptr,{},"Prototype order 1..20 (band filters have 2*order poles)."},
            text("btype",false,"lowpass",{"lowpass","highpass","bandpass","bandstop"},"Band type."),
            json("wn",true,"Critical frequency: a number, or [low, high] for bandpass/bandstop. In Hz when fs is given, else as a fraction of the Nyquist frequency (0..1). Strictly inside (0, Nyquist). For butter/cheby1/ellip/bessel(phase-normalised) it is the -3 dB / ripple edge; for cheby2 the stopband edge."),
            num("fs","Sampling frequency > 0; omit for normalised frequencies (1 = Nyquist)."),
            num("rp","Passband ripple in dB, (0, 100]; cheby1 and ellip only."),
            num("rs","Stopband attenuation in dB, (0, 200]; cheby2 and ellip only."),
            Param{"freqs",Type::Json,false,nullptr,{},"Frequencies (<=200, in the units of wn) at which to report magnitude and phase (radians) of the filter response."},
            json("x",false,"Finite signal to filter (vector). Required by the write form."),
            text("mode",false,nullptr,{"sosfilt","sosfiltfilt"},"With x. sosfilt is causal (phase lag, zero initial state); sosfiltfilt (default) runs forward and backward: zero phase, squared magnitude, needs more than 3*(2*sections+1) samples.")}});
        finish(Operation{write?"sig_spectrum_to":"sig_spectrum",write?Access::Write:Access::Read,std::string("Spectral estimates with scipy.signal: Welch / periodogram power spectral density (or spectrum), cross-spectral density and magnitude-squared coherence. One-sided, real signals. Reports f and the estimate, n_segments, df, peak_freq and total_power (the Parseval integral). Statuses OK, NUMERICAL_FAILURE.")+tail,"object",{
            text("method",true,nullptr,{"welch","periodogram","csd","coherence"},"welch averages windowed segments; periodogram is one window over the whole signal; csd needs y and returns pxy_re / pxy_im; coherence needs y and returns cxy in [0, 1]."),
            json("x",true,"Finite real signal (vector), at least 4 samples."),
            json("y",false,"Second signal of the same length (csd and coherence only)."),
            Param{"fs",Type::Number,false,1.0,{},"Sampling frequency > 0 (default 1)."},
            text("window",false,nullptr,{"hann","hamming","blackman","boxcar","bartlett","flattop","parzen","bohman","blackmanharris","nuttall","barthann","cosine","triang"},"Window (default hann; boxcar for periodogram)."),
            integer("nperseg",nullptr,"Segment length, 2..min(signal length, 65536) (default min(256, length)). Not for periodogram."),
            integer("noverlap",nullptr,"Overlap between segments, 0..nperseg-1 (default nperseg/2). Not for periodogram."),
            integer("nfft",nullptr,"FFT length (zero padding), >= nperseg (>= the signal length for periodogram), at most 131072."),
            text("detrend",false,"constant",{"constant","linear","none"},"Per-segment detrending."),
            text("scaling",false,nullptr,{"density","spectrum"},"density (default): power per Hz; spectrum: power per bin (an on-bin sine of amplitude A peaks at A^2/2). Not for coherence."),
            text("average",false,nullptr,{"mean","median"},"Segment averaging, welch and csd only (default mean).")}});
        finish(Operation{write?"sig_peaks_to":"sig_peaks",write?Access::Write:Access::Read,std::string("Local maxima of a finite signal (scipy.signal.find_peaks) with heights, prominences and widths at rel_height. Plateaus report their middle sample; the first and last sample are never peaks. Result: n_peaks, indices, heights, prominences, widths (read form: at most ~225 peaks).")+(write?" The write form stores them and answers NO_PEAKS (no artifact) when nothing qualifies.":" Use the _to form for more peaks."),"object",{
            json("x",true,"Finite signal (vector), at least 3 samples."),
            json("height",false,"Minimum height: a number or [min, max] (null for an open end)."),
            json("threshold",false,"Minimum vertical distance to both neighbours: a number or [min, max], >= 0."),
            num("distance","Minimum horizontal distance between peaks in samples, >= 1; smaller peaks are removed first."),
            json("prominence",false,"Required prominence: a number or [min, max], >= 0."),
            json("width",false,"Required width at rel_height: a number or [min, max] in samples, >= 0."),
            integer("wlen",nullptr,"Window length in samples (>= 3) for the prominence base search."),
            Param{"rel_height",Type::Number,false,0.5,{},"Height fraction of the prominence at which widths are measured, in (0, 1] (0.5 = full width at half maximum)."}}});
        finish(Operation{write?"spatial_knn_to":"spatial_knn",write?Access::Write:Access::Read,std::string("Nearest neighbours with a k-d tree (scipy.spatial.cKDTree), exact. k-nearest: indices and distances (rows sorted by distance; the index order of equal distances is implementation-defined, compare distances). Radius: counts per query plus flat indices and distances (CSR layout, sorted by distance within a query). Status OK.")+tail,"object",{
            json("points",true,"Finite n x d matrix (a vector is one-dimensional), d <= 32."),
            json("queries",false,"Finite m x d query matrix; omit to query the points themselves (a point is then its own nearest neighbour unless exclude_self)."),
            integer("k",nullptr,"Number of neighbours, 1..1000 and <= n. Exactly one of k and radius."),
            num("radius","Return every neighbour within this distance (inclusive), > 0. Exactly one of k and radius."),
            Param{"p",Type::Json,false,2,{},"Minkowski norm: a number in [1, 64] (1 Manhattan, 2 Euclidean) or \"inf\" (Chebyshev). Default 2."},
            Param{"exclude_self",Type::Boolean,false,false,{},"Only without queries: drop each point from its own neighbour list."},
            integer("max_pairs",100000,"Radius mode: refuse (OUTPUT_TOO_LARGE) when more than this many neighbour pairs would result, 1..1000000.")}});
        finish(Operation{write?"spatial_hull_to":"spatial_hull",write?Access::Write:Access::Read,std::string("Convex hull, Delaunay triangulation or Voronoi diagram with QHull (scipy.spatial). The QHull options are fixed per kind and never caller-supplied. Degenerate input (collinear/coplanar points, fewer than dim+1 points, duplicates only, a QHull error) is a domain status DEGENERATE with a message, not an error. Statuses OK, DEGENERATE, NUMERICAL_FAILURE, TIME_LIMIT. convex_hull: vertices, simplices, volume, area (in 2-D: area and perimeter); delaunay: simplices, volume, n_coplanar; voronoi: vertices, point_region, regions and ridge_vertices as flat arrays with offsets (-1 is the point at infinity), ridge_points.")+tail,"object",{
            json("points",true,"Finite n x d matrix, 2 <= d <= 6. Point caps shrink with d (convex_hull: 20000/20000/800/400/100 for d=2..6; delaunay and voronoi: 20000/800/300/100/50)."),
            text("kind",true,nullptr,{"convex_hull","delaunay","voronoi"},"Which QHull structure to compute.")}});
        finish(Operation{write?"spatial_dist_to":"spatial_dist",write?Access::Write:Access::Read,std::string("Pairwise distance matrix (scipy.spatial.distance.cdist) between the rows of a and b (b omitted: a with itself, a symmetric matrix). Metrics come from a closed list. A pair for which the metric is undefined (cosine/correlation of a zero or constant vector, an indefinite VI, overflow) makes the status UNDEFINED with a count, and no matrix is returned. Statuses OK, UNDEFINED, TIME_LIMIT. Result: shape, distances (n x m).")+tail,"object",{
            json("a",true,"Finite n x d matrix."),
            json("b",false,"Finite m x d matrix; omit to compare a with itself. n*m is limited to 1,000,000."),
            text("metric",true,nullptr,{"euclidean","sqeuclidean","cityblock","chebyshev","cosine","correlation","mahalanobis","jaccard","hamming"},"Distance. jaccard treats non-zero as true; hamming is the fraction of differing coordinates; mahalanobis needs vi."),
            json("vi",false,"mahalanobis only: the d x d symmetric inverse covariance matrix.")}});
    }
}
