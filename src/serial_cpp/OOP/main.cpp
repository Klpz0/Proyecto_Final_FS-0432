#include <iostream>
#include "grilla.hpp"
#include <sys/time.h>

int main(){
  Grilla capacitor(50,1e-5, 0.9);
  capacitor.imprimir();
  return 0;
}
