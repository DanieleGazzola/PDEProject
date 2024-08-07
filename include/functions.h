#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

using namespace dealii;

template <int dim>
class MuFunction : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
class BetaFunction : public Function<dim>
{
public:
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
class GammaFunction : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
class SourceFunction : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

#endif // FUNCTIONS_H
