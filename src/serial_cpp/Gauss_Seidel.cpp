#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>

using namespace std;

double max_abs_diff(const vector<double>& vec1, const vector<double>& vec2) {
    double max_diff = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = std::abs(vec1[i] - vec2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

void imprimir (vector<double> phi, int cm){
  for (int i = 0; i <= 10*cm; ++i) {
    for (int j = 0; j <= 10*cm; ++j) {
      cout << " " << phi[i*(10*cm+1) + j];  // alineado con ancho fijo
      }
    cout << endl;
    }
}

void capacitor(double omega,double tolerance,int tamaño){ //tamaño de la matriz
  int cm= tamaño/10; //conversion de cm a bloques de la grilla
  int dimension= (10*cm+1)*(10*cm+1);
  vector<double> phi(dimension,0);
  
  //valores frontera
  for (int i=0;i<=6*cm;i++){
    phi[((i+2*cm)*(10*cm+1)+2*cm)]=1.0;
    phi[((i+2*cm)*(10*cm+1)+8*cm)]=-1.0;  
  };
  
  //imprimir(phi,cm);
  
  vector<double> phi_copy=phi;
  double delta=1.0;
  int its=0;

  while (delta>tolerance){
    its+=1;
    for (int i=0;i<10*cm+1;i++){
      for (int j=0;j<10*cm+1;j++){
        if ((j == 2*cm || j == 8*cm) && (i >= 2*cm||i <= 8*cm)){ // ignorar frontera
	  continue;
	}
	if (i == 0 || i == 10*cm || j == 0 || j == 10*cm){
	  continue;
	}
	phi[i*(10*cm+1)+j] = (1+omega)*0.25*(phi[(i+1)*(10*cm+1)+j]+phi[(i-1)*(10*cm+1)+j]+phi[i*(10*cm+1)+j+1]+phi[i*(10*cm+1)+j-1])-omega*(phi[i*(10*cm+1)+j]);
      }
    }
    delta = max_abs_diff(phi, phi_copy);
    phi_copy=phi;
  }

  imprimir(phi,cm);
  cout<<"its: "<<its<<endl;
}

int main(){
  capacitor(0,0.0001,100);
  return 0;
}

