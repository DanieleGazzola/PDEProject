#include <fstream>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include "operator.h"
#include "setup.h"
#include "functions.h"

using namespace dealii;
using VectorType = LinearAlgebra::distributed::Vector<double>;

int main() {

// problem parameters
    const unsigned int dim = 2;
    const unsigned int fe_degree = 1;
    const unsigned int ref_level = 6;

// problem setup
    Triangulation<dim> triangulation;
    FE_Q<dim> fe(fe_degree);
    DoFHandler<dim> dof_handler;
    MappingQ1<dim> mapping;
    AffineConstraints<double> constraints;
    CustomOperator<dim, fe_degree> custom_operator;
    VectorType rhs;
    VectorType solution;

    custom_operator.clear();
    setup_problem(triangulation, fe, dof_handler, ref_level);

    constraints.clear();
    const types::boundary_id dirichlet_boundary_id = 0;
    GFunction<dim> g_function;
    VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary_id, g_function, constraints);
    constraints.close();

    MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);

    std::shared_ptr<MatrixFree<dim, double>> matrix_free_ptr(new MatrixFree<dim, double>());

    matrix_free_ptr->reinit(mapping, dof_handler, constraints, QGauss<dim>(fe.degree + 1), additional_data);
    custom_operator.initialize(matrix_free_ptr);

    custom_operator.evaluate_mu(MuFunction<dim>());
    custom_operator.evaluate_beta(BetaFunction<dim>());
    custom_operator.evaluate_gamma(GammaFunction<dim>());
    custom_operator.initialize_dof_vector(solution);
    custom_operator.initialize_dof_vector(rhs);

//assemble rhs
    SourceFunction<dim> source_function;
    rhs = 0;
    FEEvaluation<dim, fe_degree> phi(*custom_operator.get_matrix_free());

    for (unsigned int cell = 0; cell < custom_operator.get_matrix_free()->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(source_function.value(phi.quadrature_point(q)), q);
        phi.integrate(EvaluationFlags::values);
        phi.distribute_local_to_global(rhs);
      }
    rhs.compress(VectorOperation::add);
    constraints.distribute(rhs);


// solve system
    SolverControl solver_control(1000, 1e-12);
    SolverCG<VectorType> solver(solver_control);

    constraints.distribute(solution);
    solver.solve(custom_operator, solution, rhs, PreconditionIdentity());
    constraints.distribute(solution);

// output
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(rhs, "rhs");
    data_out.build_patches(mapping);

    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);

    return 0;
}
