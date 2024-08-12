#include "operator.h"

using VectorType = LinearAlgebra::distributed::Vector<double>;

template<int dim>
void CustomOperator<dim>::initialize(std::shared_ptr<const MatrixFree<dim, double>> matrix_free,
                                     const Function<dim> &mu_function,
                                     const Function<dim> &beta_function,
                                     const Function<dim> &gamma_function)
{
    const std::vector<unsigned int> row = {};
    const std::vector<unsigned int> col = {};
    this->MatrixFreeOperators::Base<dim, VectorType, VectorizedArray<double>>::initialize(matrix_free, row, col);
    mu = &mu_function;
    beta = &beta_function;
    gamma = &gamma_function;
}

template<int dim>
void CustomOperator<dim>::vmult(VectorType &dst, const VectorType &src) const
{

    FEEvaluation<dim, 1> fe_eval(*(this->data));

    for (unsigned int cell = 0; cell < (*(this->data)).n_cell_batches(); ++cell)
    {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true, true);

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
            dealii::Point<dim, double> qp_scalar;

            for (unsigned int d = 0; d < dim; ++d)
                    qp_scalar[d] = fe_eval.quadrature_point(q)[d][0];

            const auto mu_value = mu->value(qp_scalar);
            const auto beta_value = beta->gradient(qp_scalar);
            const auto gamma_value = gamma->value(qp_scalar);

            // -∇·(µ∇u)
            fe_eval.submit_gradient(-mu_value * fe_eval.get_gradient(q), q);

            // ∇·(βu)
            fe_eval.submit_gradient(beta_value * fe_eval.get_value(q), q);

            // γu
            fe_eval.submit_value(gamma_value * fe_eval.get_value(q), q);
        }

        fe_eval.integrate(true, true);
        fe_eval.distribute_local_to_global(dst);
    }
}

template class CustomOperator<2>;
template class CustomOperator<3>;
