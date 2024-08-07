#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include "operator.h"
#include "boundaries.h"
#include "setup.h"
#include "functions.h"

using namespace dealii;

int main()
{
    const unsigned int dim = 2;
    const unsigned int fe_degree = 1;
    const unsigned int ref_level = 5;

// problem setup
    Triangulation<dim> triangulation;
    FE_Q<dim> fe(fe_degree);
    DoFHandler<dim> dof_handler;

    setup_problem(triangulation, fe, dof_handler, ref_level);

// matrix free setup
    MatrixFree<dim, double> matrix_free;
    CustomOperator<dim> custom_operator;

    auto matrix_free_ptr = std::make_shared<MatrixFree<dim, double>>();

    MuFunction<dim> mu_function;
    BetaFunction<dim> beta_function;
    GammaFunction<dim> gamma_function;

    matrix_free.reinit(dof_handler);
    custom_operator.initialize(matrix_free_ptr, mu_function, beta_function, gamma_function);

// solve system
    SourceFunction<dim> source_function;

    VectorType rhs(matrix_free.get_vector_partitioner());
    VectorType solution(matrix_free.get_vector_partitioner());

    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), source_function, rhs);

    SolverControl solver_control(1000, 1e-12);
    SolverCG<VectorType> solver(solver_control);

    solver.solve(custom_operator, solution, rhs, PreconditionIdentity());

// output
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);

    return 0;
}
