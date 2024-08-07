#include "functions.h"

template <int dim>
double MuFunction<dim>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
    return 1.0; // -∇·(µ∇u)
}

template <int dim>
Tensor<1, dim> BetaFunction<dim>::gradient(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
    Tensor<1, dim> beta; // ∇·(βu)
    for(unsigned int d = 0; d < dim; ++d)
        beta[d] = 1.0;
    return beta;
}

template <int dim>
double GammaFunction<dim>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
    return 1.0; // γu
}

template <int dim>
double SourceFunction<dim>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
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
