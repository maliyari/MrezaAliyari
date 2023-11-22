#include <iostream>

#include<vector>

void print(std::vector <int>& v) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < v.size(); i++)
   std::cout << v.at(i) << ' ';

}

int main() {

   std::vector<int> v = {2,4,3,5,6};
   print(v);
   return 0;
}
