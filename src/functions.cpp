#include "functions.h"

template <int dim>
double MuFunction<dim>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
    return 0.01; // -∇·(µ∇u)
}

template <int dim>
Tensor<1, dim> BetaFunction<dim>::gradient(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
    Tensor<1, dim> beta; // ∇·(βu)
    beta[0] = 1.0;
    beta[1] = 0.5;
    return beta;
}

template <int dim>
double GammaFunction<dim>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
    return 1.0; // γu
}

template <int dim>
double SourceFunction<dim>::value(const Point<dim> & p, const unsigned int /*component*/) const
{
    return 10 * std::exp(-100 * ((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5))); // f
}

template <int dim>
double GFunction<dim>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
    return 0.0; // g
}

template class MuFunction<2>;
template class MuFunction<3>;
template class BetaFunction<2>;
template class BetaFunction<3>;
template class GammaFunction<2>;
template class GammaFunction<3>;
template class SourceFunction<2>;
template class SourceFunction<3>;
template class GFunction<2>;
template class GFunction<3>;
