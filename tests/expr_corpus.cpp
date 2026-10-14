#include "expr_parser.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
    if(argc!=2)return 2;
    std::ifstream file(argv[1]);lmcpp::Json corpus;file>>corpus;
    // Optional "dims": pycli's vector-index extension. Absent dims = the byte-identical rcli parser behaviour.
    size_t count=0;
    for(const char* group:{"accept","reject"})for(const auto& fixture:corpus[group]){
        std::set<std::string> names;
        for(const auto& name:fixture.value("names",lmcpp::Json::array()))names.insert(name.get<std::string>());
        bool reject=std::string(group)=="reject";
        try{std::map<std::string,size_t> dims;if(fixture.contains("dims"))for(auto it=fixture["dims"].begin();it!=fixture["dims"].end();++it)dims[it.key()]=it.value().get<size_t>();
        pycli_expr::Parser parser(fixture["expression"],names,fixture.value("allow_reducers",false),dims);parser.parse();if(reject)throw std::runtime_error("accepted rejected expression");}
        catch(const pycli_expr::ParseFailure& e){if(!reject||fixture["expected_tool_code"]!=e.code){std::cerr<<fixture.dump()<<" got "<<e.code<<'\n';return 1;}}
        ++count;
    }
    std::cout<<count<<" shared expression corpus fixtures passed\n";
}
