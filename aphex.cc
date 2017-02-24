
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// #include <boost/program_options.hpp>

#include <mandy/function_tools.h>

#include <mandy/elastic_problem.h>
#include <mandy/piezoelectric_problem.h>

#include <fstream>
#include <iostream>

#include <algorithm>    // std::transform
#include <functional>   // std::plus

namespace aphex
{

  template <int dim>
  class Aphex
  {
  public:

    /**
     * Class constructor.
     */
    Aphex (const std::string &prm);

    /**
     * Class destructor.
     */
    ~Aphex ();

    /**
     * Run.
     */
    void run ();
    
  private:

    /**
     * MPI communicator.
     */
    MPI_Comm mpi_communicator;
    
    /**
     * A distributed grid on which all computations are done.
     */
    dealii::parallel::distributed::Triangulation<dim> triangulation;

    /**
     * Solution of the elastic problem.
     */
    dealii::PETScWrappers::MPI::Vector displacement;

    /**
     * Solution of the piezoelectric.
     */
    dealii::PETScWrappers::MPI::Vector piezoelectric_potential;
    
    /**
     * Parallel iostream.
     */
    dealii::ConditionalOStream pcout;

    /**
     * Stop clock.
     */
    dealii::TimerOutput timer;
    
  };

  
  template <int dim>
  Aphex<dim>::Aphex (const std::string &prm)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
		   typename dealii::Triangulation<dim>::MeshSmoothing
		   (dealii::Triangulation<dim>::smoothing_on_refinement |
		    dealii::Triangulation<dim>::smoothing_on_coarsening)),
    pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process (mpi_communicator) == 0)),
    timer (mpi_communicator, pcout,
     	   dealii::TimerOutput::summary,
     	   dealii::TimerOutput::wall_times)
  {}
  

  template <int dim>
  Aphex<dim>::~Aphex ()
  {}

  template <int dim>
  void
  Aphex<dim>::run ()
  {

    try
      {

	dealii::GridGenerator::hyper_cube (triangulation, -10, 10);
	// triangulation.refine_global (parameters.get_integer ("Global mesh refinement steps"));
	triangulation.refine_global (2);

	{
	  dealii::TimerOutput::Scope time (timer, "material");
	  mandy::FunctionTools<3> material (triangulation, "material.prm");
	  material.run ();
	}

	{
	  dealii::TimerOutput::Scope time (timer, "elastic problem");
	  mandy::ElasticProblem<3> elastic_problem (triangulation, displacement,
						    mpi_communicator, "elastic.prm");
	  elastic_problem.run ();
	}

	{
	  dealii::TimerOutput::Scope time (timer, "piezoelectric problem");
	  mandy::PiezoelectricProblem<3> piezoelectric_problem (triangulation, displacement,
								mpi_communicator, "piezoelectric.prm");
	  piezoelectric_problem.run ();
	}

      }
    
    catch (std::exception &exc)
      {
	std::cerr << std::endl << std::endl
		  << "----------------------------------------------------"
		  << std::endl;
	std::cerr << "Exception on processing: " << std::endl
		  << exc.what() << std::endl
		  << "Aborting!" << std::endl
		  << "----------------------------------------------------"
		  << std::endl;
      }
    
    catch (...)
      {
	std::cerr << std::endl << std::endl
		  << "----------------------------------------------------"
		  << std::endl;
	std::cerr << "Unknown exception!" << std::endl
		  << "Aborting!" << std::endl
		  << "----------------------------------------------------"
		  << std::endl;
      }
    
  } // run ()
  
} // namespace aphex

/**
 * Main function: Initialise problem and run it.
 */
int main (int argc, char *argv[])
{
  
  // Initialise MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, 1);

  try
    {
#ifdef INCLUDED_EXTERNAL_BOOST
      // Parse the commandline.
      boost::program_options::options_description description {"Options"};
      description.add_options ()
	("help", "Help screen")
	("prm", value<string> ()->default_value ("aphex.prm"), "Parameter file");

      boost::program_options::variables_map vmap;
      boost::program_options::store
	(boost::program_options::parse_command_line (argc, argv, description), vmap);

      if (vmap.count ("help"))
	std::cout << description
		  << std::endl;

      else if (vmap.count ("prm"))
	std::cout << "Parameter file: " << vmap["prm"].as<string>
		  << std::endl;

      // else if (...)
#endif      
      
      aphex::Aphex<3> aphex ("material.prm");
      aphex.run ();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
