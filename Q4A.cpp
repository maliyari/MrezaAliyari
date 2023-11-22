#include <iostream>
#include <bits/stdc++.h>
using namespace std;

bool isprime(int n)
{
    bool result=true;
    // Corner case
    if (n <= 1){
       result=false;
    }


    // Check from 2 to n-1
    for (int i = 2; i < n; i++)
        if (n % i == 0){
          result=false;
            break;
        }

    if (result==true){
     cout<<"Number is prime ";
    } else{
     cout<<"Number is not prime ";
    }

    return result;

}
   void test_isprime() {
        std::cout << "isprime(2) = " << isprime(2) <<'\n';
        std::cout << "isprime(10) = " << isprime(10) <<'\n';
        std::cout << "isprime(17) = " << isprime(17) <<'\n';
}

// test above function
int main()
{
    test_isprime();
    return 0;
}

