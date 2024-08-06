#include "setup.h"

template<int dim>
void setup_problem(
    Triangulation<dim> &triangulation,
    FE_Q<dim> &fe,
    DoFHandler<dim> &dof_handler,
    const unsigned int level)
{
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(level);

    dof_handler.initialize(triangulation, fe);
}

template void setup_problem<2>(Triangulation<2> &, FE_Q<2> &, DoFHandler<2> &, const unsigned int);
template void setup_problem<3>(Triangulation<3> &, FE_Q<3> &, DoFHandler<3> &, const unsigned int);