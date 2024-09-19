#ifndef SETUP_H
#define SETUP_H

#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>

using namespace dealii;

template<int dim>
void setup_problem(
    Triangulation<dim> &triangulation,
    FE_Q<dim>          &fe,
    DoFHandler<dim>    &dof_handler,
    const unsigned int level);

#endif // SETUP_H
