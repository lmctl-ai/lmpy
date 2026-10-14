// Exercise the actual CLI parsers under ASan/UBSan with structure-aware ZIP mutations.
#define main pycli_entry_for_fuzz
#include "../src/main.cpp"
#undef main
#include <random>
#include <iostream>
void put(std::string& b,size_t p,uint64_t v,size_t n){for(size_t i=0;i<n;++i)b[p+i]=char(v>>(8*i));}
std::string seed(){std::string h="{'descr': '|u1', 'fortran_order': False, 'shape': (1,), }\n",n="__pycli_table__.npy";std::string body="{\"version\":1,\"n_rows\":0,\"columns\":[\"x\"]}";h="{'descr': '|u1', 'fortran_order': False, 'shape': ("+std::to_string(body.size())+",), }\n";std::string a="\x93NUMPY\x01";a.push_back(0);a.resize(10);put(a,8,h.size(),2);a+=h;a+=body;std::string b(30,0);put(b,0,0x04034b50,4);put(b,18,a.size(),4);put(b,22,a.size(),4);put(b,26,n.size(),2);b+=n;b+=a;size_t c=b.size();b.resize(c+46);put(b,c,0x02014b50,4);put(b,c+20,a.size(),4);put(b,c+24,a.size(),4);put(b,c+28,n.size(),2);b+=n;size_t e=b.size();b.resize(e+22);put(b,e,0x06054b50,4);put(b,e+8,1,2);put(b,e+10,1,2);put(b,e+12,e-c,4);put(b,e+16,c,4);return b;}
int main(){std::mt19937_64 rng(2026);auto good=seed();frames::npz(good);size_t accepted=0,rejected=0;for(int i=0;i<10000;++i){auto b=good;size_t c=good.find("PK\x01\x02");switch(i%5){case 0:break;case 1:{const int fields[]={8,10,20,24,28,30,32,34,42};put(b,c+fields[rng()%9],rng(),2);break;}case 2:b[rng()%b.size()]=char(rng());break;case 3:b.resize(rng()%b.size());break;case 4:put(b,b.size()-6,rng(),4);break;}try{frames::npz(b);++accepted;}catch(const lmcpp::Error&){++rejected;}}
if(accepted<2000||rejected<1000)return 1;
for(int i=0;i<10000;++i){try{frames::csv(std::string(rng()%100,char(rng())),",",100);}catch(const lmcpp::Error&){}try{frames::rule(std::to_string(rng())+"D");}catch(const lmcpp::Error&){}Json node={{"col","x"},{"op",i%2?"eq":"contains"},{"value",std::string(rng()%100,char(rng()))}};frames::predicate(node);}
std::cout<<"table fuzz: "<<accepted<<" ZIPs passed all size/header/layout checks; "<<rejected<<" rejected; 10000 CSV/rule/predicate cases\n";
}
