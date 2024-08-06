#include "operator.h"

template<int dim>
void CustomOperator<dim>::initialize(const MatrixFree<dim, double> &matrix_free,
                                     const Function<dim> &mu_function,
                                     const Function<dim> &beta_function,
                                     const Function<dim> &gamma_function)
{
    this->MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<double>, VectorizedArray<double>>::initialize(matrix_free);
    mu = &mu_function;
    beta = &beta_function;
    gamma = &gamma_function;
}

template<int dim>
void CustomOperator<dim>::apply_add(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const
{

    FEEvaluation<dim, 1> fe_eval(this->data);

    for (unsigned int cell = 0; cell < this->data.n_macro_cells(); ++cell)
    {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true, true);

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
            const auto mu_value = mu->value(fe_eval.quadrature_point(q));
            const auto beta_value = beta->value(fe_eval.quadrature_point(q));
            const auto gamma_value = gamma->value(fe_eval.quadrature_point(q));

            // -∇·(µ∇u)
            fe_eval.submit_gradient(-mu_value * fe_eval.get_gradient(q), q);

            // ∇·(βu)
            fe_eval.submit_value(fe_eval.get_gradient(q) * beta_value, q);

            // γu
            fe_eval.submit_value(gamma_value * fe_eval.get_value(q), q);
        }

        fe_eval.integrate(true, true);
        fe_eval.distribute_local_to_global(dst);
    }
}

template class CustomOperator<2>;
template class CustomOperator<3>;
