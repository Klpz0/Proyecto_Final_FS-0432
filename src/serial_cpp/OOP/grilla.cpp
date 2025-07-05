#include <iostream>
#include <math.h>
#include "grilla.hpp"
#include <vector>
#include <fstream>

#include <iomanip> // para usar fixed y setprecision
#include <sys/time.h>
#include <omp.h>



Grilla::Grilla(){
  tamaño=100;
  omega=0.0;
  tolerancia=0.001; // tolerancia 1e-3
  dimension=101*101;
  num_iteraciones=0;
}

Grilla::Grilla(int l, double t, double w){
  tamaño=l;
  omega=w;
  tolerancia=t;
  dimension=(l+1)*(l+1);
  num_iteraciones=0;
}

Grilla::~Grilla(){
}

std::vector<double> Grilla::potencial() {
  std::vector<double>phi(dimension,0);
  int cm2=0.2*tamaño;
  int cm8=0.8*tamaño; 
  for (int i=cm2;i<=cm8;i++){      //valores iniciales
    phi[(i*(tamaño+1)+cm2)]=1.0;
    phi[(i*(tamaño+1)+cm8)]=-1.0;
  };
  std::vector<double> phi_copy=phi;
  double delta=1.0;
  num_iteraciones=0;
  while (delta>tolerancia){
    num_iteraciones+=1;
    for (int i=0;i<tamaño+1;i++){
      for (int j=0;j<tamaño+1;j++){
        if ((j == cm2 || j == cm8) && (i >= cm2 && i <= cm8)){ // ignorar frontera
          continue;
        }
        if (i == 0 || i == tamaño || j == 0 || j == tamaño){
          continue;
        }
        phi[i*(tamaño+1)+j] = (1+omega)*0.25*(phi[(i+1)*(tamaño+1)+j]+phi[(i-1)*(tamaño+1)+j]+phi[i*(tamaño+1)+j+1]+phi[i*(tamaño+1)+j-1])-omega*(phi[i*(tamaño+1)+j]);
      }
    }
    delta = max_abs_diff(phi, phi_copy);
    phi_copy = phi;
  }

  return phi;
}

double Grilla::max_abs_diff(const std::vector<double>& vec1, const std::vector<double>& vec2){
  double max_diff = 0.0f;
  for (size_t i = 0; i < vec1.size(); ++i) {
    double diff = std::abs(vec1[i] - vec2[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}
 

int Grilla::iteraciones() {
  return num_iteraciones;
}

void Grilla::imprimir() {
  std::ofstream file("matriz.txt");
  std::vector<double> phi=potencial();
  for (int i = 0; i <= tamaño; ++i) {
    for (int j = 0; j <= tamaño; ++j) {
      file << " " << phi[i*(tamaño+1) + j];  // alineado con ancho fijo
      }
    file << std::endl;
    }
  file.close();
}
