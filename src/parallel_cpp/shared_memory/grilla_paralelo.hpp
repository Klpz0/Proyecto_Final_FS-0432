#ifndef GRILLA_PARALELO_HPP
#define GRILLA_PARALELO_HPP

#include <vector>

class Grilla
{
  public:
    void imprimir();
    int iteraciones();
    ~Grilla(); // Destructor
    Grilla(int l, double t, double w); // Custom constructor
    Grilla(); 
  private:
    int tamaño;
    double tolerancia;
    double omega;
    int dimension;
    int num_iteraciones;
    std::vector<double> potencial();
    double max_abs_diff(const std::vector<double>& vec1, const std::vector<double>& vec2);
};
#endif
