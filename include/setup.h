#ifndef SETUP_H
#define SETUP_H

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

using namespace dealii;

template<int dim>
void setup_problem(
    parallel::fullydistributed::Triangulation<dim> &mesh,
    FE_Q<dim>                                      &fe,
    DoFHandler<dim>                                &dof_handler,
    const unsigned int                             level
);

#endif // SETUP_H
