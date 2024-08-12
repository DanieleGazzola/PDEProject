#ifndef OPERATOR_H
#define OPERATOR_H

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;
using VectorType = LinearAlgebra::distributed::Vector<double>;

template<int dim>
class CustomOperator : public MatrixFreeOperators::Base<dim, VectorType, VectorizedArray<double>>
{
public:
    CustomOperator() = default;

    void initialize(std::shared_ptr<const MatrixFree<dim, double>> matrix_free,
                    const Function<dim> &mu,
                    const Function<dim> &beta,
                    const Function<dim> &gamma);

    void vmult(VectorType &dst, const VectorType &src) const;

    virtual void compute_diagonal() override {}
    virtual void apply_add(VectorType & /*dst*/, const VectorType & /*src*/) const override {}


private:
    SmartPointer<const Function<dim>> mu;
    SmartPointer<const Function<dim>> beta;
    SmartPointer<const Function<dim>> gamma;
};

#endif // OPERATOR_H
