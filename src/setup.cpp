#include "setup.h"

template<int dim>
void setup_problem(
    Triangulation<dim> &triangulation,
    FE_Q<dim>          &fe,
    DoFHandler<dim>    &dof_handler,
    const unsigned int level)
{
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(level);

    // dim = 2
    for (auto &cell : triangulation.active_cell_iterators())
    {
        if (cell->at_boundary())
        {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
                if (cell->face(f)->at_boundary()) 
                {
                    if (std::abs(cell->face(f)->center()[0] - 0.0) < 1e-12)
                        cell->face(f)->set_boundary_id(0); // Left boundary -> g
                    else if (std::abs(cell->face(f)->center()[1] - 0.0) < 1e-12)
                        cell->face(f)->set_boundary_id(0); // Bottom boundary -> g
                    else
                        cell->face(f)->set_boundary_id(0); // Other boundaries -> h TODO
                }
            }
        }
    }

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(fe);
}

template<int dim>
void print_dof(
    DoFHandler<dim> &dof_handler,
    std::string     filename)
{
    std::cout << std::endl;
    std::cout << "Saving mesh to " << filename << std::endl;
    std::cout << std::endl;
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    std::ofstream output(filename);
    data_out.build_patches();
    data_out.write_vtk(output);
}

template void setup_problem<2>(Triangulation<2> &, FE_Q<2> &, DoFHandler<2> &, const unsigned int);
template void setup_problem<3>(Triangulation<3> &, FE_Q<3> &, DoFHandler<3> &, const unsigned int);
template void print_dof<2>(DoFHandler<2> &, std::string);
template void print_dof<3>(DoFHandler<3> &, std::string);