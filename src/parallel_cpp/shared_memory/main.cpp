#include <iostream>
#include "grilla_paralelo.hpp"
#include <sys/time.h>

double seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;

  return sec;
}

int main(){
  Grilla capacitor(50,1e-5, 0.9);
  double time1=seconds();
  capacitor.imprimir();
  double time2=seconds();
  std::cout << "Tiempo: " << time2-time1 << std::endl;
  return 0;
}
