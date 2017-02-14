


#include <deal.II/base/function.h>
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

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>


namespace mandy
{

  /**
   * Solve the equations of linear elasticity on a finite element
   * grid.
   */
  template <int dim>
  class LinearElasticity
  {
  public:

    /**
     * Class constructor.
     */
    LinearElasticity (const std::string &prm_file);

    /**
     * Class destructor.
     */
    ~LinearElasticity ();

    /**
     * Wrapper function, that controls the order of excecution.
     */
    void run ();
    
  private:

    /**
     * Make intial coarse grid.
     */
    void make_coarse_grid ();

    /**
     * Setup system matrices and vectors.
     */
    void setup_system ();
    
    /**
     * MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /**
     * A distributed grid on which all computations are done.
     */
    dealii::parallel::distributed::Triangulation<dim> triangulation;

    /**
     * Scalar DoF handler primarily used for interpolating material
     * identification.
     */
    dealii::DoFHandler<dim> dof_handler;

    // /**
    //  * Scalar valued finite element primarily used for interpolating
    //  * material iudentification.
    //  */
    dealii::FESystem<dim> fe;

    /**
     * Index set of locally owned DoFs.
     */
    dealii::IndexSet locally_owned_dofs;

    /**
     * Index set of locally relevant DoFs.
     */
    dealii::IndexSet locally_relevant_dofs;

    /**
     * A list of (hanging node) constraints.
     */
    dealii::ConstraintMatrix constraints;
    
    /**
     * System matrix - a mass matrix.
     */
    dealii::PETScWrappers::MPI::SparseMatrix system_matrix;

    /**
     * Locally relevant solution vector.
     */
    dealii::PETScWrappers::MPI::Vector locally_relevant_solution;

    /**
     * System right hand side function - interpolated function.
     */
    dealii::PETScWrappers::MPI::Vector system_rhs;
    
    /**
     * Parallel iostream.
     */
    dealii::ConditionalOStream pcout;

    /**
     * Stop clock.
     */
    dealii::TimerOutput timer;
    
    /**
     * Input parameter file.
     */
    dealii::ParameterHandler parameters;
    
  }; // LinearElasticity

  
  /**
   * Class constructor.
   */
  template <int dim>
  LinearElasticity<dim>::LinearElasticity (const std::string &prm_file)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename dealii::Triangulation<dim>::MeshSmoothing
                   (dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening)),
    dof_handler (triangulation),
    fe (dealii::FE_Q<dim> (2), 1),
    // Other initialisations go here.
    pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process (mpi_communicator) == 0)),
    timer (mpi_communicator, pcout,
	   dealii::TimerOutput::summary,
	   dealii::TimerOutput::wall_times)
  {
    parameters.declare_entry ("Global mesh refinement steps", "5",
                              dealii::Patterns::Integer (0, 20),
                              "The number of times the 1-cell coarse mesh should "
                              "be refined globally for our computations.");

    parameters.declare_entry ("MaterialID", "0",
                              dealii::Patterns::Anything (),
                              "A functional description of the material ID.");
    
    parameters.parse_input (prm_file);
  }

  
  /**
   * Class destructor.
   */
  template <int dim>
  LinearElasticity<dim>::~LinearElasticity ()
  {
    // Wipe DoF handlers.
    dof_handler.clear ();
  }


  /**
   * Make initial coarse grid.
   */
  template <int dim>
  void
  LinearElasticity<dim>::make_coarse_grid ()
  {
    dealii::TimerOutput::Scope time (timer, "make coarse grid");

    // Create a coarse grid according to the parameters given in the
    // input file.
    dealii::GridGenerator::hyper_cube (triangulation, -10, 10);
    
    triangulation.refine_global (parameters.get_integer ("Global mesh refinement steps"));
  }


  /**
   * Setup system matrices and vectors.
   */
  template <int dim>
  void LinearElasticity<dim>::setup_system ()
  {
    dealii::TimerOutput::Scope time (timer, "setup system");

    // Determine locally relevant DoFs.
    dof_handler.distribute_dofs (fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    dealii::DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);

    // Initialise distributed vectors.
    locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs,
				      mpi_communicator);
    system_rhs.reinit (locally_owned_dofs,
		       mpi_communicator);

    // Setup hanging node constraints.
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    constraints.close ();

    // Finally, create a distributed sparsity pattern and initialise
    // the system matrix from that.
    dealii::DynamicSparsityPattern dsp (locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
    dealii::SparsityTools::distribute_sparsity_pattern (dsp,
							dof_handler.n_locally_owned_dofs_per_processor (),
							mpi_communicator,
							locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs, locally_owned_dofs,
                          dsp, mpi_communicator);

  }
  
  /**
   * Run the application in the order specified.
   */
  template <int dim>
  void
  LinearElasticity<dim>::run ()
  {
    const unsigned int n_cycles = 1;
    
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

	if (cycle==0)
	  make_coarse_grid ();

	pcout << "MaterialID: "
	      << std::endl
	      << "   Number of active cells:       "
	      << triangulation.n_global_active_cells ()
	      << std::endl
	      << "   Number of degrees of freedom: "
	      << dof_handler.n_dofs ()
	      << std::endl;

	setup_system ();
	
      }     // for cycle<n_cycles
  } 
  
} // namespace mandy


/**
 * Main function: Initialise problem and run it.
 */
int main (int argc, char *argv[])
{
  // Initialise MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, 1);
  
  try
    {
      mandy::LinearElasticity<2> linear_elasticity ("step-4.prm");
      linear_elasticity.run ();
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
