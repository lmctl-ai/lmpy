#pragma once
#include <lmcpp/lmcpp.hpp>
#include "expr_spec.hpp"
#include <cerrno>
#include <cstdlib>
#include <map>
#include <set>

namespace pycli_expr {
using lmcpp::Json;
struct ParseFailure { std::string code; size_t position; std::string near; };

class Parser {
 public:
  // pycli extension (not in the shared rcli spec): names listed in `dims` are vectors that may only appear as
  // name[<unsigned integer literal>] with the literal below their length. With empty `dims` this is rcli's parser.
  Parser(std::string text, std::set<std::string> names, bool allow_reducers, std::map<std::string,size_t> dims = {})
      : text_(std::move(text)), names_(std::move(names)), allow_reducers_(allow_reducers), dims_(std::move(dims)), spec_(Json::parse(PYCLI_EXPR_SPEC_JSON)) {}

  Json parse() {
    if (text_.size() > cap("bytes")) fail("EXPR_TOO_LONG", cap("bytes"));
    for (size_t i=0;i<text_.size();++i) {
      if (text_.compare(i,3,"<<-")==0 || text_.compare(i,2,"<-")==0 || text_.compare(i,2,"->")==0) fail("EXPR_SYNTAX",i);
    }
    for (size_t i=0; i<text_.size(); ++i) {
      const unsigned char c=static_cast<unsigned char>(text_[i]);
      if (!(c=='\t'||c=='\n'||(c>=0x20&&c<=0x7e))) fail("EXPR_BAD_BYTE", i);
    }
    auto result=parse_sum(0);
    whitespace();
    if (pos_!=text_.size()) fail("EXPR_SYNTAX",pos_);
    if (tokens_>cap("tokens")) fail("EXPR_TOO_LONG",pos_);
    if (nodes_>cap("nodes")) fail("EXPR_TOO_MANY_NODES",pos_);
    if (identifiers_.size()>cap("idents")) fail("EXPR_TOO_MANY_NODES",pos_);
    return result;
  }
  const std::set<std::string>& identifiers() const { return identifiers_; }

 private:
  std::string text_; std::set<std::string> names_; bool allow_reducers_; std::map<std::string,size_t> dims_;
  Json spec_; size_t pos_=0, tokens_=0, nodes_=0, calls_=0;
  std::set<std::string> identifiers_;
  size_t cap(const char *name) const { return spec_["caps"][name].get<size_t>(); }
  void whitespace(){while(pos_<text_.size()&&(text_[pos_]==' '||text_[pos_]=='\t'||text_[pos_]=='\n'))++pos_;}
  bool starts(const std::string&s) const { return text_.compare(pos_,s.size(),s)==0; }
  void token(){++tokens_;if(tokens_>cap("tokens"))fail("EXPR_TOO_LONG",pos_);}
  Json node(Json x){++nodes_;if(nodes_>cap("nodes"))fail("EXPR_TOO_MANY_NODES",pos_);if(x.value("t",std::string())=="op"){bool found=false;for(const auto&o:spec_["operators"])found|=o.get<std::string>()==x["o"].get<std::string>();if(!found)fail("EXPR_SPEC_MISMATCH",pos_);}return x;}
  [[noreturn]] void fail(const std::string&code,size_t at) const {
    size_t from=at>8?at-8:0;size_t end=std::min(text_.size(),from+16);
    auto continuation=[&](size_t p){return p<text_.size()&&(static_cast<unsigned char>(text_[p])&0xc0)==0x80;};
    while(from<end&&continuation(from))++from;
    if(end<text_.size())while(end>from&&continuation(end))--end;
    throw ParseFailure{code,at,text_.substr(from,end-from)};
  }
  bool take(char c){whitespace();if(pos_<text_.size()&&text_[pos_]==c){++pos_;token();return true;}return false;}
  bool takeop(const std::string&op){whitespace();if(starts(op)){pos_+=op.size();token();return true;}return false;}
  void enter(size_t depth){if(depth>cap("depth"))fail("EXPR_TOO_DEEP",pos_);}
  Json parse_sum(size_t d){enter(d);auto lhs=parse_product(d);for(;;){if(take('+'))lhs=node(Json{{"t","op"},{"o","+"},{"a",Json::array({lhs,parse_product(d)})}});else if(take('-'))lhs=node(Json{{"t","op"},{"o","-"},{"a",Json::array({lhs,parse_product(d)})}});else return lhs;}}
  Json parse_product(size_t d){auto lhs=parse_unary(d);for(;;){if(take('*'))lhs=node(Json{{"t","op"},{"o","*"},{"a",Json::array({lhs,parse_unary(d)})}});else if(take('/'))lhs=node(Json{{"t","op"},{"o","/"},{"a",Json::array({lhs,parse_unary(d)})}});else return lhs;}}
  Json parse_unary(size_t d){enter(d);if(take('-'))return node(Json{{"t","neg"},{"a",Json::array({parse_unary(d+1)})}});return parse_power(d);}
  Json parse_power(size_t d){auto lhs=parse_atom(d);if(take('^'))return node(Json{{"t","op"},{"o","^"},{"a",Json::array({lhs,parse_unary(d+1)})}});return lhs;}
  Json parse_rel(size_t d){auto lhs=parse_sum(d);for(const std::string op:{"<=",">=","==","!=","<",">"})if(takeop(op))return node(Json{{"t","op"},{"o",op},{"a",Json::array({lhs,parse_sum(d)})}});return lhs;}
  Json parse_condition(size_t d){auto lhs=parse_or(d);return lhs;}
  Json parse_or(size_t d){auto lhs=parse_and(d);while(take('|'))lhs=node(Json{{"t","op"},{"o","|"},{"a",Json::array({lhs,parse_and(d)})}});return lhs;}
  Json parse_and(size_t d){auto lhs=parse_rel(d);while(take('&'))lhs=node(Json{{"t","op"},{"o","&"},{"a",Json::array({lhs,parse_rel(d)})}});return lhs;}
  Json parse_atom(size_t d){
    enter(d);whitespace();if(pos_>=text_.size())fail("EXPR_SYNTAX",pos_);
    if(take('(')){auto x=parse_sum(d+1);if(!take(')'))fail("EXPR_SYNTAX",pos_);return x;}
    const size_t start=pos_;
    if(text_[pos_]>='0'&&text_[pos_]<='9'){
      while(pos_<text_.size()&&text_[pos_]>='0'&&text_[pos_]<='9')++pos_;
      if(pos_<text_.size()&&text_[pos_]=='.'){++pos_;while(pos_<text_.size()&&text_[pos_]>='0'&&text_[pos_]<='9')++pos_;}
      if(pos_<text_.size()&&(text_[pos_]=='e'||text_[pos_]=='E')){++pos_;if(pos_<text_.size()&&(text_[pos_]=='+'||text_[pos_]=='-'))++pos_;size_t es=pos_;while(pos_<text_.size()&&text_[pos_]>='0'&&text_[pos_]<='9')++pos_;if(es==pos_)fail("EXPR_SYNTAX",start);}
      token();std::string lit=text_.substr(start,pos_-start);char*end=nullptr;double v=std::strtod(lit.c_str(),&end);if(!end||*end||!std::isfinite(v))fail("EXPR_SYNTAX",start);return node(Json{{"t","num"},{"v",v}});
    }
    if(text_[pos_]>='a'&&text_[pos_]<='z'){
      ++pos_;while(pos_<text_.size()&&((text_[pos_]>='a'&&text_[pos_]<='z')||(text_[pos_]>='0'&&text_[pos_]<='9')||text_[pos_]=='_'))++pos_;
      token();std::string name=text_.substr(start,pos_-start);if(name.size()>32)fail("EXPR_SYNTAX",start);
      whitespace();if(pos_<text_.size()&&text_[pos_]=='(')return parse_call(name,d+1,start);
      if(spec_["constants"].contains(name))return node(Json{{"t","const"},{"n",name}});
      if(spec_["functions"].contains(name))fail("EXPR_NAME_COLLISION",start);
      for(const auto&reserved:spec_.value("reserved",Json::array()))if(reserved.get<std::string>()==name)fail("EXPR_NAME_COLLISION",start);
      if(!names_.count(name))fail("EXPR_UNKNOWN_IDENT",start);
      identifiers_.insert(name);
      auto vec=dims_.find(name);
      if(vec!=dims_.end()){
        whitespace();if(pos_>=text_.size()||text_[pos_]!='[')fail("EXPR_VECTOR_INDEX",start);
        ++pos_;token();whitespace();
        const size_t digits=pos_;while(pos_<text_.size()&&text_[pos_]>='0'&&text_[pos_]<='9')++pos_;
        if(pos_==digits||pos_-digits>6||(text_[digits]=='0'&&pos_-digits>1))fail("EXPR_INDEX",digits);
        const size_t index=std::stoul(text_.substr(digits,pos_-digits));
        whitespace();if(pos_>=text_.size())fail("EXPR_SYNTAX",pos_);
        if(text_[pos_]!=']'||index>=vec->second)fail("EXPR_INDEX",digits);
        ++pos_;token();return node(Json{{"t","idx"},{"n",name},{"i",index}});
      }
      return node(Json{{"t","var"},{"n",name}});
    }
    fail("EXPR_SYNTAX",pos_);
  }
  Json parse_call(const std::string&name,size_t d,size_t start){
    if(!spec_["functions"].contains(name))fail("EXPR_UNKNOWN_FUNCTION",start);
    if(spec_["functions"][name].value("reducer",false)&&!allow_reducers_)fail("EXPR_NOT_SCALAR",start);
    if(++calls_>cap("calls"))fail("EXPR_TOO_DEEP",start);enter(d);
    if(!take('('))fail("EXPR_SYNTAX",pos_);
    if(name=="ifelse"){
      auto c=parse_condition(d);if(!c.is_object()||c.value("t",std::string())!="op"||!std::set<std::string>{"<","<=",">",">=","==","!=","&","|"}.count(c.value("o",std::string())))fail("EXPR_SYNTAX",start);if(!take(','))fail("EXPR_ARITY",pos_);auto yes=parse_sum(d);if(!take(','))fail("EXPR_ARITY",pos_);auto no=parse_sum(d);if(!take(')'))fail("EXPR_SYNTAX",pos_);--calls_;
      return node(Json{{"t","ifelse"},{"c",c},{"a",yes},{"b",no}});
    }
    Json args=Json::array();if(!take(')')){for(;;){args.push_back(parse_sum(d));if(take(')'))break;if(!take(','))fail("EXPR_SYNTAX",pos_);}}
    --calls_;if(args.size()!=spec_["functions"][name]["arity"].get<size_t>())fail("EXPR_ARITY",start);
    return node(Json{{"t","call"},{"f",name},{"a",args}});
  }
};
}
