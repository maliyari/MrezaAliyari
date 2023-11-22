// Program to print all prime factors
#include <iostream>
# include <stdio.h>
# include <math.h>
#include <vector>
using namespace std;

// A function to print all prime factors of a given number n
std::vector<int> prime_factorize(int n) {
    std::vector<int> answer;

    // Print the number of 2s that divide n
    while (n%2 == 0)
    {
        answer.push_back(2);
        n = n/2;
    }

    // n must be odd at this point.  So we can skip
    // one element (Note i = i +2)
    for (int i = 3; i <= sqrt(n); i = i+2)
    {
        // While i divides n, print i and divide n
        while (n%i == 0)
        {
            answer.push_back(i);
            n = n/i;
        }
    }

    // This condition is to handle the case when n
    // is a prime number greater than 2
    if (n > 2)
        answer.push_back(n);
    return answer;
}

/* test above function */

void print_vector(std::vector <int> v) {
   std::cout << "The prime factorize vector elements are : "<<endl;

   for(int i=0; i < v.size(); i++)
   std::cout << v.at(i) << ' '<<endl;
}

void test_prime_factorize() {
    print_vector(prime_factorize(2));
    print_vector(prime_factorize(72));
    print_vector(prime_factorize(196));
}

int main()
{
    test_prime_factorize();
}
