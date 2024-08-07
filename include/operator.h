#ifndef OPERATOR_H
#define OPERATOR_H

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

template<int dim>
class CustomOperator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<double>, VectorizedArray<double>>
{
public:
    CustomOperator() = default;

    void initialize(std::shared_ptr<const MatrixFree<dim, double>> matrix_free,
                    const Function<dim> &mu,
                    const Function<dim> &beta,
                    const Function<dim> &gamma);

    void apply_add(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const override;

private:
    SmartPointer<const Function<dim>> mu;
    SmartPointer<const Function<dim>> beta;
    SmartPointer<const Function<dim>> gamma;
};

#endif // OPERATOR_H
