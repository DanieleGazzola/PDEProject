#ifndef OPERATOR_H
#define OPERATOR_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "functions.h"

using namespace dealii;
using VectorType = LinearAlgebra::distributed::Vector<double>;

template<int dim, int fe_degree>
class CustomOperator : public MatrixFreeOperators::Base<dim, VectorType>
{
public:

    CustomOperator();

    void clear() override;

    void evaluate_mu(const MuFunction<dim> &mu_function);

    void evaluate_beta(const BetaFunction<dim> &beta_function);

    void evaluate_gamma(const GammaFunction<dim> &gamma_function);

    virtual void compute_diagonal() override;

    virtual void apply_add(VectorType       &dst,
                           const VectorType &src)
                           const override;

    void local_apply(const MatrixFree<dim, double>               &data,
                     VectorType                                  &dst, 
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &cell_range)
                     const;

private:

    Table<2, VectorizedArray<double>> mu_coefficients;
    Table<3, VectorizedArray<double>> beta_coefficients;
    Table<2, VectorizedArray<double>> gamma_coefficients;

};

#endif // OPERATOR_H
