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
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include "operator.h"
#include "setup.h"
#include "functions.h"

using namespace dealii;
using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int dim, int fe_degree, int ref_level>
void run_simulation_matrix_free(ConditionalOStream &pcout){

    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    parallel::fullydistributed::Triangulation<dim> mesh(MPI_COMM_WORLD);
    FE_Q<dim> fe(fe_degree);
    DoFHandler<dim> dof_handler;
    MappingQ1<dim> mapping;
    QGauss<dim> quad(fe.degree + 1);
    IndexSet locally_owned_dofs;
    AffineConstraints<double> constraints;
    CustomOperator<dim, fe_degree> custom_operator;
    VectorType rhs;
    VectorType solution;

    const auto start_time = std::chrono::high_resolution_clock::now();

    custom_operator.clear();
    setup_problem(mesh, fe, dof_handler, ref_level);
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    constraints.clear();
    const types::boundary_id dirichlet_boundary_id = 0;
    GFunction<dim> g_function;
    VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary_id, g_function, constraints);
    constraints.close();

/*
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
    additional_data.mapping_update_flags_boundary_faces = (update_values | update_JxW_values | update_quadrature_points);
    additional_data.mapping_update_flags_inner_faces = (update_values | update_JxW_values | update_quadrature_points);
    additional_data.mapping_update_flags_faces_by_cells  = (update_values | update_JxW_values | update_quadrature_points);
    additional_data.hold_all_faces_to_owned_cells = true;

    std::shared_ptr<MatrixFree<dim, double>> matrix_free_ptr(new MatrixFree<dim, double>());

    matrix_free_ptr->reinit(mapping, dof_handler, constraints, quad, additional_data);
    custom_operator.initialize(matrix_free_ptr);

    custom_operator.evaluate_mu(MuFunction<dim>());
    custom_operator.evaluate_beta(BetaFunction<dim>());
    custom_operator.evaluate_gamma(GammaFunction<dim>());
    custom_operator.initialize_dof_vector(solution);
    custom_operator.initialize_dof_vector(rhs);

    const auto setup_time = std::chrono::high_resolution_clock::now();

    SourceFunction<dim> source_function;
    HFunction<dim> h_function;
    //const types::boundary_id neumann_boundary_id = 1;

    rhs = 0;
    FEEvaluation<dim, fe_degree> phi(*custom_operator.get_matrix_free());
    FEFaceEvaluation<dim, fe_degree> phi_face(*custom_operator.get_matrix_free());

    for (unsigned int cell = 0; cell < custom_operator.get_matrix_free()->n_cell_batches(); ++cell)
    {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(source_function.value(phi.quadrature_point(q)), q);
        phi.integrate(EvaluationFlags::values);
        phi.distribute_local_to_global(rhs);
    }

    for (unsigned int cell = 0; cell < custom_operator.get_matrix_free()->n_cell_batches(); ++cell){
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face){
            phi_face.reinit(cell, face);
            if (phi_face.at_boundary() && phi_face.boundary_id() == neumann_boundary_id){
                for (unsigned int q = 0; q < phi_face.n_q_points; ++q)
                    phi_face.submit_value(h_function.value(phi_face.quadrature_point(q)), q);
                phi_face.integrate(EvaluationFlags::values);
                phi_face.distribute_local_to_global(rhs);
            }
        }
    }
  
    rhs.compress(VectorOperation::add);
    constraints.distribute(rhs);

    const auto rhs_time = std::chrono::high_resolution_clock::now();

    SolverControl solver_control(100000, 1e-12);
    SolverCG<VectorType> solver(solver_control);

    solver.solve(custom_operator, solution, rhs, PreconditionIdentity());
    constraints.distribute(solution);

    const auto end_time = std::chrono::high_resolution_clock::now();

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(rhs, "rhs");
    data_out.build_patches(mapping);

    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);

    pcout << "Setup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(setup_time - start_time).count() << " ms" << std::endl;
    pcout << "RHS assembly time: " << std::chrono::duration_cast<std::chrono::milliseconds>(rhs_time - setup_time).count() << " ms" << std::endl;
    pcout << "Solve time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - rhs_time).count() << " ms" << std::endl;
*/
}

template <int dim, int fe_degree, int ref_level>
void run_simulation_classic(ConditionalOStream & /*pcout*/){
    

    return;
}

int main(int argc, char *argv[]){

    const unsigned int dim       = 2;
    const unsigned int fe_degree = 1;
    const unsigned int ref_level = 6;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    ConditionalOStream pcout(std::cout, mpi_rank == 0);

    pcout << "Running matrix-free simulation" << std::endl;
    run_simulation_matrix_free<dim, fe_degree, ref_level>(pcout);

    pcout << "Running classic simulation" << std::endl;
    run_simulation_classic<dim, fe_degree, ref_level>(pcout);

    return 0;
}