#include "functions.h"

template <int dim>
double MuFunction<dim>::value(const Point<dim> &p, const unsigned int component) const
{
    return 1.0; // -∇·(µ∇u)
}

template <int dim>
Tensor<1, dim> BetaFunction<dim>::value(const Point<dim> &p) const
{
    Tensor<1, dim> beta; // ∇·(βu)
    beta[0] = 1.0;
    return beta;
}

template <int dim>
double GammaFunction<dim>::value(const Point<dim> &p, const unsigned int component) const
{
    return 1.0; // γu
}

template <int dim>
double SourceFunction<dim>::value(const Point<dim> &p, const unsigned int component) const
{
    return 1.0; // f
}

template class MuFunction<2>;
template class MuFunction<3>;
template class BetaFunction<2>;
template class BetaFunction<3>;
template class GammaFunction<2>;
template class GammaFunction<3>;
template class SourceFunction<2>;
template class SourceFunction<3>;
