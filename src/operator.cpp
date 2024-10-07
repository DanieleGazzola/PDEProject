#include "operator.h"

using VectorType = LinearAlgebra::distributed::Vector<double>;

template<int dim, int fe_degree>
CustomOperator<dim, fe_degree>::CustomOperator(const AffineConstraints<double> &constraints) : MatrixFreeOperators::Base<dim, VectorType>() {
    this->mu_coefficients.reinit(0, 0);
    this->beta_coefficients.reinit({0, 0, 0});
    this->gamma_coefficients.reinit(0, 0);
    this->constraints_ptr = &constraints;
}

template<int dim, int fe_degree>
void CustomOperator<dim, fe_degree>::clear() {
    mu_coefficients.reinit(0, 0);
    beta_coefficients.reinit({0, 0, 0});
    gamma_coefficients.reinit(0, 0);
    MatrixFreeOperators::Base<dim, VectorType>::clear();
}

template<int dim, int fe_degree>
void CustomOperator<dim, fe_degree>::evaluate_mu(const MuFunction<dim> &mu_function) {

    const unsigned int n_cells = this->data->n_cell_batches();
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi(*this->data);
    mu_coefficients.reinit(n_cells, phi.n_q_points);

    for (unsigned int cell = 0; cell < n_cells; ++cell){
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
            mu_coefficients(cell, q) = mu_function.value(phi.quadrature_point(q));
    }
}

template<int dim, int fe_degree>
void CustomOperator<dim, fe_degree>::evaluate_beta(const BetaFunction<dim> &beta_function) {

    const unsigned int n_cells = this->data->n_cell_batches();
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi(*this->data);
    beta_coefficients.reinit({n_cells, phi.n_q_points, dim});

    for (unsigned int cell = 0; cell < n_cells; ++cell){
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
            for (unsigned int d = 0; d < dim; ++d)
                beta_coefficients(cell, q, d) = beta_function.value(phi.quadrature_point(q), d);
        }
    }
}

template<int dim, int fe_degree>
void CustomOperator<dim, fe_degree>::evaluate_gamma(const GammaFunction<dim> &gamma_function) {

    const unsigned int n_cells = this->data->n_cell_batches();
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi(*this->data);
    gamma_coefficients.reinit(n_cells, phi.n_q_points);

    for (unsigned int cell = 0; cell < n_cells; ++cell){
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
            gamma_coefficients(cell, q) = gamma_function.value(phi.quadrature_point(q));
    }
}

template<int dim, int fe_degree>
void CustomOperator<dim, fe_degree>::apply_add(VectorType &dst, const VectorType &src) const {
    this->data->cell_loop(&CustomOperator::local_apply, this, dst, src);
}

template<int dim, int fe_degree>
void CustomOperator<dim, fe_degree>::local_apply(const MatrixFree<dim, double>               &data, 
                                                 VectorType                                  &dst, 
                                                 const VectorType                            &src, 
                                                 const std::pair<unsigned int, unsigned int> &cell_range) const {

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.read_dof_values_plain(src);
        
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
            phi.submit_gradient(mu_coefficients(cell, q) * phi.get_gradient(q), q);

            VectorizedArray<double> sum = 0;
            for (unsigned int d = 0; d < dim; ++d)
                sum += beta_coefficients(cell, q, d) * phi.get_gradient(q)[d];

            phi.submit_value((gamma_coefficients(cell, q) - sum) * phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
    }
}


template<int dim, int fe_degree>
void CustomOperator<dim, fe_degree>::compute_diagonal() {}

template class CustomOperator<2, 1>;
template class CustomOperator<3, 1>;
