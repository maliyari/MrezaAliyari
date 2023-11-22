#include<iostream>
#include<vector>
using namespace std;

std::vector<int> factorize(int num) {
    std::vector<int> answer;
    int j;
    for(j=1; j <=num; j++) {
      if (num % j == 0)
      answer.push_back(j);

   }
    return answer;
}


void print_vector(std::vector <int> v) {
   std::cout << "The factorize vector elements are : "<<endl;

   for(int i=0; i < v.size(); i++)
   std::cout << v.at(i) << ' '<<endl;
}

void test_factorize() {
    print_vector(factorize(2));
    print_vector(factorize(72));
    print_vector(factorize(196));
}


int main() {

   test_factorize();

   return 0;
}
