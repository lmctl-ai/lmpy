// S8b-2 graph front end: graph_paths and graph_flow (+ _to twins). Edges come from a stored table (`frame`, columns named explicitly) or inline
// `records`; no graph file format is ever read. Node classes, node membership, caps (edges, distinct nodes, all-pairs matrix size) and output
// budgets are decided here, before any process starts.
#pragma once
constexpr size_t G_EDGES=100000, G_NODES=100000, G_FLOW_EDGES=20000, G_FLOW_NODES=20000, G_INLINE_NODES=450, G_ALLPAIRS_NODES=5000, G_MATRIX=1900000, G_RECORD_CELLS=1000;
bool graph_op(const std::string& op){return op=="graph_paths"||op=="graph_paths_to"||op=="graph_flow"||op=="graph_flow_to";}
struct GraphIn{bool ints=true;std::set<int64_t> inodes;std::set<std::string> snodes;size_t rows=0;bool has_negative=false;};
bool graph_given(const Params& in,const char* k){return in.contains(k)&&!in[k].is_null();}
size_t graph_nodes(const GraphIn& g){return g.inodes.size()+g.snodes.size();}
// min over the non-missing values of an int64/float64 column (NaN ignored), read straight from the stored members
double s8_column_min(const S8Table& t,const std::string& c){
    size_t i=0;while(t.meta["columns"][i]!=c)++i;
    auto raw=s8_member(t,"c"+std::to_string(i)),mask=s8_member(t,"m"+std::to_string(i));
    const std::string d=s8_dtype(t,c);double low=INFINITY;
    for(size_t r=0;r<raw.rows;++r){
        if(t.bytes[mask.offset+r])continue;
        double v;if(d=="float64")std::memcpy(&v,t.bytes.data()+raw.offset+r*8,8);else{int64_t k;std::memcpy(&k,t.bytes.data()+raw.offset+r*8,8);v=double(k);}
        if(!std::isnan(v))low=std::min(low,v);
    }
    return low;
}
GraphIn read_graph(const Params& in,S8Extra& x,size_t max_edges,size_t max_nodes,const std::vector<std::string>& numeric){
    GraphIn g;const bool framed=graph_given(in,"frame"),inline_records=graph_given(in,"records");
    if(framed==inline_records)bad_param("give the edges as exactly one of frame (a .npz table) and records (inline objects)");
    for(const char* k:{"from_column","to_column"})if(!graph_given(in,k)||!in[k].is_string())bad_param(std::string(k)+" must name the endpoint column");
    const std::string from=in["from_column"],to=in["to_column"];
    if(from==to)bad_param("from_column and to_column must be different columns");
    if(framed){
        S8Table t=s8_table(in,"frame");s8_need_column(t,from);s8_need_column(t,to);
        const std::string a=s8_dtype(t,from),b=s8_dtype(t,to);
        auto klass=[&](const std::string& d,const std::string& c){if(d=="int64")return std::string("int");if(d=="string"||d=="category")return std::string("string");invalid("node column "+c+" has dtype "+d+": node ids are int64, string or category","BAD_DTYPE");return std::string();};
        const std::string ka=klass(a,from),kb=klass(b,to);
        if(ka!=kb)invalid("the endpoint columns mix integer and string node ids ("+a+" / "+b+")","BAD_DTYPE");
        g.ints=ka=="int";g.rows=s8_rows(t);
        if(g.rows<1)invalid("the edge table has no rows","BAD_DATA");
        if(g.rows>max_edges)invalid("the graph may have at most "+std::to_string(max_edges)+" edges (got "+std::to_string(g.rows)+")","INPUT_TOO_LARGE");
        for(const auto& c:numeric){s8_need_column(t,c);const std::string d=s8_dtype(t,c);if(d!="int64"&&d!="float64")invalid("column "+c+" has dtype "+d+": weights, capacities and costs are int64 or float64","BAD_DTYPE");}
        for(const char* c:{from.c_str(),to.c_str()}){
            size_t i=0;while(t.meta["columns"][i]!=c)++i;
            auto raw=s8_member(t,"c"+std::to_string(i)),mask=s8_member(t,"m"+std::to_string(i));
            const std::string dtype=s8_dtype(t,c);
            if(raw.rows!=g.rows||mask.rows!=g.rows||mask.dtype!="|b1")invalid("table column shape/mask mismatch","BAD_DATA");
            if((dtype=="int64"||dtype=="category")&&raw.dtype!="<i8")invalid("table dtype mismatch","BAD_DATA");
            if(dtype=="string"&&raw.dtype.rfind("|S",0)!=0)invalid("table dtype mismatch","BAD_DATA");
            for(size_t r=0;r<g.rows;++r){
                if(t.bytes[mask.offset+r])invalid(std::string("endpoint column ")+c+" has a missing value at row "+std::to_string(r),"NA_NOT_ALLOWED");
                const char* value=t.bytes.data()+raw.offset+r*raw.width;
                if(dtype=="string"){std::string v(value,raw.width);auto z=v.find('\0');if(z!=std::string::npos)v.resize(z);g.snodes.insert(v);}
                else if(dtype=="category"){int64_t k;std::memcpy(&k,value,8);const auto& levels=t.meta["column_meta"][c]["categories"];if(!levels.is_array()||k<0||uint64_t(k)>=levels.size())invalid("invalid category codes","BAD_DATA");g.snodes.insert(levels[k].get<std::string>());}
                else{int64_t k;std::memcpy(&k,value,8);g.inodes.insert(k);}
                if(graph_nodes(g)>max_nodes)invalid("the graph may have at most "+std::to_string(max_nodes)+" distinct nodes","INPUT_TOO_LARGE");
            }
        }
        for(const auto& c:numeric)if(s8_column_min(t,c)<0)g.has_negative=true;
        x.tables["frame"]=std::move(t.bytes);
    }else{
        const Json& records=in["records"];
        if(!records.is_array()||records.empty())invalid("records must be a nonempty list of objects","BAD_DATA");
        if(records.size()*(2+numeric.size())>G_RECORD_CELLS)invalid("inline records are limited to 1000 cells: write a table with frame_import for larger graphs","BAD_DATA");
        g.rows=records.size();
        std::vector<std::string> columns={from,to};for(const auto& c:numeric)columns.push_back(c);
        bool first=true;
        if(!records[0].is_object())invalid("every record is an object","BAD_DATA");
        for(const auto& c:columns)if(!records[0].contains(c))invalid("unknown column: "+c,"UNKNOWN_COLUMN");
        for(const auto& r:records){
            if(!r.is_object())invalid("every record is an object","BAD_DATA");
            for(const auto& c:columns)if(!r.contains(c)||r[c].is_null())invalid("a record lacks the column "+c,"BAD_DATA");
            bool a_int=r[from].is_number_integer(),b_int=r[to].is_number_integer(),a_str=r[from].is_string(),b_str=r[to].is_string();
            if(!((a_int&&b_int)||(a_str&&b_str)))invalid("node ids are integers or strings, the same kind in both endpoint columns","BAD_DTYPE");
            if(first){g.ints=a_int;first=false;}else if(g.ints!=a_int)invalid("records mix integer and string node ids","BAD_DTYPE");
            if(g.ints){g.inodes.insert(r[from].get<int64_t>());g.inodes.insert(r[to].get<int64_t>());}else{g.snodes.insert(r[from].get<std::string>());g.snodes.insert(r[to].get<std::string>());}
            for(const auto& c:numeric){if(!r[c].is_number())invalid("column "+c+" must be numeric","BAD_DTYPE");if(r[c].get<double>()<0)g.has_negative=true;}
        }
        if(graph_nodes(g)>max_nodes)invalid("the graph may have at most "+std::to_string(max_nodes)+" distinct nodes","INPUT_TOO_LARGE");
    }
    return g;
}
void graph_node(const Json& v,const GraphIn& g,const char* what){
    if(g.ints){
        if(!v.is_number_integer())bad_param(std::string(what)+" must be an integer node id (the graph's node ids are integers)");
        if(!g.inodes.count(v.get<int64_t>()))invalid(std::string(what)+" "+v.dump()+" is not a node of the graph","UNKNOWN_NODE");
    }else{
        if(!v.is_string())bad_param(std::string(what)+" must be a string node id (the graph's node ids are strings)");
        if(!g.snodes.count(v.get<std::string>()))invalid(std::string(what)+" "+v.dump()+" is not a node of the graph","UNKNOWN_NODE");
    }
}
Json prepare_graph_paths(const Params& in,bool twin,S8Extra& x){
    Json p=in;
    const std::string method=in["method"];const bool directed=in["directed"];
    const bool weighted=graph_given(in,"weight_column");
    if(method!="dijkstra"&&method!="bellman_ford"&&method!="bfs")bad_param("method is dijkstra, bellman_ford or bfs");
    if(method=="bfs"&&weighted)bad_param("bfs counts edges: leave weight_column out (or use dijkstra / bellman_ford)");
    GraphIn g=read_graph(in,x,G_EDGES,G_NODES,weighted?std::vector<std::string>{in["weight_column"].get<std::string>()}:std::vector<std::string>{});
    const bool all_pairs=in["all_pairs"];
    if((method=="dijkstra"||graph_given(in,"k_shortest"))&&g.has_negative)bad_param("dijkstra and k_shortest need non-negative weights; use method bellman_ford for negative weights");
    const size_t n=graph_nodes(g);
    if(all_pairs){
        if(!twin)bad_param("all_pairs is a file result: use graph_paths_to");
        for(const char* k:{"source","target","k_shortest"})if(graph_given(in,k))bad_param(std::string(k)+" is not used with all_pairs");
        if(!g.ints)bad_param("all_pairs stores a numeric matrix: it needs integer node ids");
        if(n>G_ALLPAIRS_NODES)invalid("all_pairs allows at most "+std::to_string(G_ALLPAIRS_NODES)+" nodes (got "+std::to_string(n)+")","INPUT_TOO_LARGE");
        if(n*n>G_MATRIX)output_too_large("the all-pairs matrix would hold "+std::to_string(n*n)+" numbers for "+std::to_string(n)+" nodes (limit "+std::to_string(G_MATRIX)+", about 1377 nodes)");
        p["out"]=output_path(in);x.tensor_write=true;
    }else{
        if(!graph_given(in,"source"))bad_param("give the source node (or all_pairs with graph_paths_to)");
        graph_node(in["source"],g,"source");
        if(graph_given(in,"target"))graph_node(in["target"],g,"target");
        if(graph_given(in,"k_shortest")){
            if(!graph_given(in,"target"))bad_param("k_shortest needs a target");
            int64_t k=in["k_shortest"];if(k<1||k>10)bad_param("k_shortest must be 1..10");
        }
        if(!twin&&!graph_given(in,"target")&&n>G_INLINE_NODES)output_too_large("listing the distances of "+std::to_string(n)+" nodes exceeds the inline budget (limit "+std::to_string(G_INLINE_NODES)+" nodes): give a target or use graph_paths_to");
        if(twin){
            if(graph_given(in,"target")||graph_given(in,"k_shortest"))bad_param("the write form lists every node: target and k_shortest are for graph_paths");
            s8_table_out(in,x);
        }
    }
    p["node_kind"]=g.ints?"int":"string";p["n_nodes"]=n;
    return p;
}
Json prepare_graph_flow(const Params& in,bool twin,S8Extra& x){
    Json p=in;const std::string kind=in["kind"];
    if(kind!="max_flow"&&kind!="min_cut"&&kind!="min_cost_flow")bad_param("kind is max_flow, min_cut or min_cost_flow");
    if(!graph_given(in,"capacity_column")||!in["capacity_column"].is_string())bad_param("capacity_column names the capacity column");
    const bool costed=kind=="min_cost_flow";
    if(costed&&!graph_given(in,"cost_column"))bad_param("min_cost_flow needs cost_column");
    if(!costed&&graph_given(in,"cost_column"))bad_param("cost_column is only used by min_cost_flow");
    std::vector<std::string> numeric={in["capacity_column"].get<std::string>()};if(costed)numeric.push_back(in["cost_column"].get<std::string>());
    GraphIn g=read_graph(in,x,G_FLOW_EDGES,G_FLOW_NODES,numeric);
    if(costed){
        const bool pairs=graph_given(in,"supplies"),simple=graph_given(in,"source")||graph_given(in,"sink")||graph_given(in,"flow_value");
        if(pairs==simple)bad_param("give either supplies, or source, sink and flow_value");
        if(pairs){
            const Json& s=in["supplies"];if(!s.is_array()||s.empty()||s.size()>1000)bad_param("supplies is a list of 1..1000 [node, amount] pairs (positive supply, negative demand)");
            std::set<std::string> seen;
            for(const auto& pair:s){
                if(!pair.is_array()||pair.size()!=2||!pair[1].is_number_integer())bad_param("each supply is [node, integer amount]");
                graph_node(pair[0],g,"a supplies node");if(!seen.insert(pair[0].dump()).second)bad_param("a node appears twice in supplies");
            }
        }else{
            if(!graph_given(in,"source")||!graph_given(in,"sink")||!graph_given(in,"flow_value"))bad_param("source, sink and flow_value go together");
            graph_node(in["source"],g,"source");graph_node(in["sink"],g,"sink");if(in["source"]==in["sink"])bad_param("source and sink must differ");
            if(!in["flow_value"].is_number_integer()||in["flow_value"].get<int64_t>()<1)bad_param("flow_value must be a positive integer");
        }
    }else{
        for(const char* k:{"supplies","flow_value"})if(graph_given(in,k))bad_param(std::string(k)+" is only used by min_cost_flow");
        if(!graph_given(in,"source")||!graph_given(in,"sink"))bad_param(kind+" needs source and sink");
        graph_node(in["source"],g,"source");graph_node(in["sink"],g,"sink");if(in["source"]==in["sink"])bad_param("source and sink must differ");
    }
    if(twin){s8_table_out(in,x);}
    p["node_kind"]=g.ints?"int":"string";p["n_nodes"]=graph_nodes(g);
    return p;
}
Json prepare_graph(const std::string& op,const Params& in,S8Extra& x){
    const bool twin=s7_twin(op);const std::string base=twin?op.substr(0,op.size()-3):op;
    return base=="graph_paths"?prepare_graph_paths(in,twin,x):prepare_graph_flow(in,twin,x);
}

void add_graph_ops(CliSpec& spec){
    auto str=[](const char* name,bool required,Json def,std::vector<Json> values,const char* text){return Param{name,Type::String,required,std::move(def),std::move(values),text};};
    auto json=[](const char* name,bool required,const char* text){return Param{name,Type::Json,required,nullptr,{},text};};
    auto edges=[&](std::vector<Param>& params){
        params.push_back(str("frame",false,nullptr,{},"Stored .npz table with one row per edge (frame_import). Give exactly one of frame and records; graph file formats are never accepted."));
        params.push_back(Param{"records",Type::Array,false,nullptr,{},"Inline edges: objects keyed by the column names, at most 1000 cells.","",false,false,false,{},Type::Object});
        params.push_back(str("from_column",true,nullptr,{},"Column with the edge start node: int64 (integer ids), or string / category (string ids); a missing value is NA_NOT_ALLOWED."));
        params.push_back(str("to_column",true,nullptr,{},"Column with the edge end node, of the same kind as from_column."));
    };
    const char* table_out="New .npz table artifact (CONFLICT if it exists).";
    for(bool write:{false,true}){
        std::vector<Param> params;edges(params);
        params.push_back(str("weight_column",false,nullptr,{},"Edge length column (int64 or float64, finite); omitted: every edge has length 1. Not with method bfs. Parallel edges keep the lightest; self loops are ignored."));
        params.push_back(Param{"directed",Type::Boolean,false,true,{},"true: edges run from_column -> to_column; false: each edge can be used both ways."});
        params.push_back(str("method",false,"dijkstra",{"dijkstra","bellman_ford","bfs"},"dijkstra needs non-negative weights (BAD_PARAM otherwise); bellman_ford accepts negative weights and reports NEGATIVE_CYCLE; bfs counts edges."));
        params.push_back(json("source",false,"Start node (an integer, or a string for string-id graphs; it must occur in the edges, UNKNOWN_NODE otherwise). Required unless all_pairs."));
        if(!write){params.push_back(json("target",false,"End node: returns one shortest path (UNREACHABLE when there is none). Without a target the distances of all reachable nodes are listed (at most 450 nodes inline).")); }
        else params.push_back(Param{"all_pairs",Type::Boolean,false,false,{},"true: the n x n distance matrix of an integer-id graph (n <= 5000, and n*n <= 1,900,000 numbers) in `out`: tensor distance (NaN = unreachable) in the order of tensor nodes (ascending ids); bellman_ford uses Johnson's algorithm. false: a node table in `out`."});
        if(!write){params.push_back(Param{"k_shortest",Type::Integer,false,nullptr,{},"With a target: the k <= 10 shortest simple paths (Yen), non-negative weights, ascending length.",});params.push_back(Param{"all_pairs",Type::Boolean,false,false,{},"Must be false: the all-pairs matrix is a file result (graph_paths_to)."});}
        else params.push_back(str("out",true,nullptr,{},"New output artifact: a .npz node table (columns node, distance, hops, predecessor; unreachable nodes have missing distance/hops/predecessor) or, with all_pairs, a .safetensors file (distance, nodes)."));
        spec.operations.push_back(Operation{write?"graph_paths_to":"graph_paths",write?Access::Write:Access::Read,std::string("Shortest paths on a graph given as an edge table (networkx for one source; scipy.sparse.csgraph for the all-pairs matrix). Statuses OK, UNREACHABLE (the target cannot be reached; no path), NEGATIVE_CYCLE (bellman_ford found a negative cycle reachable from the source, or anywhere for all_pairs; `cycle` lists it), TIME_LIMIT. Unreachable nodes are never given a distance. Caps: 100000 edges, 100000 nodes, k_shortest <= 10, all pairs n <= 5000 and n*n <= 1,900,000 (checked before any work).")+(write?" Successful writes return artifacts[name] descriptors.":" Inline results hold at most ~900 numbers."),"object",params});
    }
    for(bool write:{false,true}){
        std::vector<Param> params;edges(params);
        params.push_back(str("capacity_column",true,nullptr,{},"Edge capacity column (int64 or float64, 0 <= c <= 1e15). Edges are directed; parallel edges add up (max_flow, min_cut) or are DUPLICATE_EDGE (min_cost_flow)."));
        params.push_back(str("cost_column",false,nullptr,{},"min_cost_flow only: integer cost per unit of flow (negative costs allowed)."));
        params.push_back(str("kind",true,nullptr,{"max_flow","min_cut","min_cost_flow"},"max_flow: value and flow edges; min_cut: value, the cut edges and the source side of a minimum cut (max-flow = min-cut); min_cost_flow: total cost and flow edges for the given supplies."));
        params.push_back(json("source",false,"max_flow, min_cut: the source node; min_cost_flow: with sink and flow_value instead of supplies."));
        params.push_back(json("sink",false,"The sink node (must differ from the source)."));
        params.push_back(Param{"flow_value",Type::Integer,false,nullptr,{},"min_cost_flow with source and sink: the amount to route, a positive integer."});
        params.push_back(Param{"supplies",Type::Json,false,nullptr,{},"min_cost_flow: an array of [node, amount] pairs (1..1000), positive = supply, negative = demand, integers; the amounts must sum to zero (UNBALANCED otherwise)."});
        if(write)params.push_back(str("out",true,nullptr,{},table_out));
        spec.operations.push_back(Operation{write?"graph_flow_to":"graph_flow",write?Access::Write:Access::Read,std::string("Network flows on a directed edge table with capacities (networkx Dinitz and network simplex). Integer capacities are exact (`integral`: true, tolerance 0); with float capacities the flow is checked for conservation and capacity within `tolerance` = 1e-9 * max(1, largest capacity) and NUMERICAL_FAILURE is reported beyond it. The returned flow is always re-verified against the edge list. Statuses OK, INFEASIBLE (the supplies cannot be routed within the capacities), UNBALANCED (supplies do not sum to zero), UNBOUNDED, NUMERICAL_FAILURE, TIME_LIMIT. Results: value / cost, flow_from, flow_to, flow (edges with positive flow; at most 300 inline), cut_from, cut_to, cut_capacity and source_side (min_cut; at most 300 each inline). Caps: 20000 edges, 20000 nodes. The write form stores the flow edges (from, to, flow) or cut edges (from, to, capacity) as a node-id table in `out`.")+(write?" Successful writes return artifacts[name] descriptors.":""),"object",params});
    }
}
