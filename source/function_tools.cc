
#include <mandy/function_tools.h>

#include <fstream>
#include <iostream>

#include <algorithm>    // std::transform
#include <functional>   // std::plus

namespace mandy
{
  
  /**
   * Class constructor.
   */
  template <int dim>
  FunctionTools<dim>::FunctionTools (dealii::parallel::distributed::Triangulation<dim> &triangulation,
				     const std::string                                 &prm)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (&triangulation),
    dof_handler (triangulation),
    fe (dealii::FE_Q<dim> (2), 1),
    // ---
    pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process (mpi_communicator) == 0)),
    timer (mpi_communicator, pcout,
     	   dealii::TimerOutput::summary,
     	   dealii::TimerOutput::wall_times)
  {
    parameters.declare_entry ("Global mesh refinement steps", "5",
                              dealii::Patterns::Integer (0, 20),
                              "The number of times the 1-cell coarse mesh should "
                              "be refined globally for our computations.");

    parameters.declare_entry ("MaterialFunction", "0",
                              dealii::Patterns::Anything (),
                              "A functional description of the inclusion material.");
    
    parameters.parse_input (prm);
  }

  
  /**
   * Class destructor.
   */
  template <int dim>
  FunctionTools<dim>::~FunctionTools ()
  {
    // Wipe DoF handlers.
    dof_handler.clear ();
  }


  /**
   * Setup system matrices and vectors.
   */
  template <int dim>
  void FunctionTools<dim>::setup_system ()
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
   * Assemble system matrices and vectors.
   *
   * TODO Ideally, we would use a function like this:
   *
   * dealii::MatrixCreator::create_mass_matrix (dof_handler, quadrature_rule, system_matrix, 1, constraints);
   *
   * however no such thing currently exists in the deal.II library for
   * parallel matrices and vectors. Instead, the mass matrix and right
   * hand side vector are assembled by hand in functions defined in
   * the namepsaces, mandy::MatrixCreator and mandy::VectorCreator,
   * respectively.
   */
  template <int dim>
  void
  FunctionTools<dim>::assemble_system ()
  {
    dealii::TimerOutput::Scope time (timer, "assemble system");

    // Define quadrature rule to be used.
    const dealii::QGauss<dim> quadrature_formula (3);

    // Initialise the function parser.
    dealii::FunctionParser<dim> material_identification;
    material_identification.initialize (dealii::FunctionParser<dim>::default_variable_names (),
					parameters.get ("MaterialFunction"),
					typename dealii::FunctionParser<dim>::ConstMap ());
    
    mandy::MatrixCreator::create_mass_matrix<dim> (fe, dof_handler, quadrature_formula,
						   system_matrix, constraints,
						   mpi_communicator);

    mandy::VectorCreator::create_right_hand_side_vector<dim> (fe, dof_handler, quadrature_formula,
     							      system_rhs, constraints,
							      material_identification,
							      mpi_communicator);
  }
  

  /**
   * Solve the linear algebra system.
   */
  template <int dim>
  unsigned int
  FunctionTools<dim>::solve ()
  {
    dealii::TimerOutput::Scope time (timer, "solve");
    
    dealii::PETScWrappers::MPI::Vector completely_distributed_solution (locally_owned_dofs, mpi_communicator);

    // Solve using conjugate gradient method with no preconditioner
    // (ie., system_matrix is ignored).
    dealii::SolverControl solver_control (dof_handler.n_dofs (), 1e-06);
    dealii::PETScWrappers::SolverCG solver (solver_control, mpi_communicator);
    dealii::PETScWrappers::PreconditionNone preconditioner (system_matrix);
    
    solver.solve (system_matrix, completely_distributed_solution, system_rhs,
		  preconditioner);
    
    // Ensure that all ghost elements are also copied as necessary.
    constraints.distribute (completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;

    // Return the number of iterations (last step) of the solve.
    return solver_control.last_step ();
  }


  /**
   * Output results, ie., finite element functions and derived
   * quantitites for this cycle..
   */
  template <int dim>
  void
  FunctionTools<dim>::output_results (const unsigned int cycle)
  {
    dealii::TimerOutput::Scope time (timer, "output_results");

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (locally_relevant_solution, "material");

    dealii::Vector<float> subdomain ((*triangulation).n_active_cells ());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain (i) = (*triangulation).locally_owned_subdomain ();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();
    
    const std::string filename = ("material-" +
                                  dealii::Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  dealii::Utilities::int_to_string
                                  ((*triangulation).locally_owned_subdomain (), 4));

    std::ofstream output ((filename + ".vtu").c_str ());
    data_out.write_vtu (output);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 1)
      {
	std::vector<std::string> filenames;
	
	for (unsigned int i=0;
	     i<dealii::Utilities::MPI::n_mpi_processes (mpi_communicator);
	     ++i)
	  filenames.push_back ("material-" +
			       dealii::Utilities::int_to_string (cycle, 2) +
			       "." +
			       dealii::Utilities::int_to_string (i, 4) +
			       ".vtu");
	std::ofstream master_output (("material-" +
				      dealii::Utilities::int_to_string (cycle, 2) +
				      ".pvtu").c_str ());

	data_out.write_pvtu_record (master_output, filenames);
      }
  }


  /**
   * Refine grid based on Kelly's error estimator working on the
   * material id (solution vector).
   */
  template <int dim>
  void FunctionTools<dim>::refine_grid ()
  {
    dealii::TimerOutput::Scope time (timer, "refine grid");

    dealii::Vector<float> estimated_error_per_cell ((*triangulation).n_active_cells());
    
    dealii::KellyErrorEstimator<dim>::estimate (dof_handler, dealii::QGauss<dim-1>(4),
						typename dealii::FunctionMap<dim>::type (),
						locally_relevant_solution,
						estimated_error_per_cell);

    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number ((*triangulation),
				       estimated_error_per_cell,
				       0.250, 0.025);

    (*triangulation).execute_coarsening_and_refinement ();
  }
  
  
  /**
   * Run the application in the order specified.
   */
  template <int dim>
  void
  FunctionTools<dim>::run ()
  {
    const unsigned int n_cycles = 3;
    
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        pcout << "FunctionTools:: Cycle " << cycle << ':'
	      << std::endl;

	if (cycle!=0)
	  refine_grid ();
	
	pcout << "   Number of active cells:       "
	      << (*triangulation).n_global_active_cells ()
	      << std::endl;

	setup_system ();

	pcout << "   Number of degrees of freedom: "
	      << dof_handler.n_dofs ()
	      << std::endl;
	
	assemble_system ();

	const unsigned int n_iterations = solve ();

	pcout << "   Solved in " << n_iterations
	      << " iterations."
	      << std::endl;

	pcout << "   Linfty-norm:                  "
	      << locally_relevant_solution.linfty_norm ()
	      << std::endl;

	// Output results if the number of processes is less than or
	// equal to 32.
	if (dealii::Utilities::MPI::n_mpi_processes (mpi_communicator) <= 32)
	  output_results (cycle);

	// timer.print_summary ();
        pcout << std::endl;
	
      } // for cycle<n_cycles
  } 
  
} // namespace mandy

template class mandy::FunctionTools<3>;
