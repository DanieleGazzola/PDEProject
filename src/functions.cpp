#include "functions.h"

template <int dim>
double MuFunction<dim>::value(const Point<dim> & p, const unsigned int component) const
{
    return value<double>(p, component);
}

template <int dim>
template <typename number>
number MuFunction<dim>::value(const Point<dim, number> & /*p*/, const unsigned int /*component*/) const
{
    return 0.1; // -∇·(µ∇u)
}

template <int dim>
double BetaFunction<dim>::value(const Point<dim> & p, const unsigned int component) const
{
    return value<double>(p, component);
}

template <int dim>
template <typename number>
number BetaFunction<dim>::value(const Point<dim, number> & /*p*/, const unsigned int component) const
{
    if(component == 0) // ∇·(βu)
        return 0.0;
    else
        return 0.0;
}

template <int dim>
double GammaFunction<dim>::value(const Point<dim> & p, const unsigned int component) const
{
    return value<double>(p, component);
}

template <int dim>
template <typename number>
number GammaFunction<dim>::value(const Point<dim, number> & /*p*/, const unsigned int /*component*/) const
{
    return 0.1; // γu
}

template <int dim>
double SourceFunction<dim>::value(const Point<dim> & p, const unsigned int component) const
{
    return value<double>(p, component);
}

template <int dim>
template <typename number>
number SourceFunction<dim>::value(const Point<dim, number> & p, const unsigned int /*component*/) const
{
    return p[0] * p[1]; // f
}

template <int dim>
double GFunction<dim>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
    return 0.1; // g
}

template <int dim>
double HFunction<dim>::value(const Point<dim> & p, const unsigned int component) const
{
    return value<double>(p, component);
}

template <int dim>
template <typename number>
number HFunction<dim>::value(const Point<dim, number> & /*p*/, const unsigned int /*component*/) const
{
    return 0.1; // h
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
template class HFunction<2>;
template class HFunction<3>;

template dealii::VectorizedArray<double> MuFunction<2>::value<dealii::VectorizedArray<double>>(const dealii::Point<2, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> BetaFunction<2>::value<dealii::VectorizedArray<double>>(const dealii::Point<2, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> GammaFunction<2>::value<dealii::VectorizedArray<double>>(const dealii::Point<2, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> SourceFunction<2>::value<dealii::VectorizedArray<double>>(const dealii::Point<2, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> HFunction<2>::value<dealii::VectorizedArray<double>>(const dealii::Point<2, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> MuFunction<3>::value<dealii::VectorizedArray<double>>(const dealii::Point<3, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> BetaFunction<3>::value<dealii::VectorizedArray<double>>(const dealii::Point<3, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> GammaFunction<3>::value<dealii::VectorizedArray<double>>(const dealii::Point<3, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> SourceFunction<3>::value<dealii::VectorizedArray<double>>(const dealii::Point<3, dealii::VectorizedArray<double>> &, const unsigned int) const;
template dealii::VectorizedArray<double> HFunction<3>::value<dealii::VectorizedArray<double>>(const dealii::Point<3, dealii::VectorizedArray<double>> &, const unsigned int) const;