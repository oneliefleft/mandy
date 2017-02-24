#include <mandy/piezoelectric_problem.h>

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
  PiezoelectricProblem<dim>::PiezoelectricProblem (dealii::parallel::distributed::Triangulation<dim> &triangulation,
						   dealii::PETScWrappers::MPI::Vector                &locally_relevant_displacement,
						   MPI_Comm                                          &mpi_communicator,
						   const std::string                                 &prm)
    :
    mpi_comm (mpi_communicator),
    triangulation (&triangulation),
    scalar_dof_handler (triangulation),
    vector_dof_handler (triangulation),
    scalar_finite_element (dealii::FE_Q<dim> (2), 1),
    vector_finite_element (dealii::FE_Q<dim> (2), dim),
    locally_relevant_displacement (&locally_relevant_displacement),
    // ---
    pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process (mpi_comm) == 0)),
    timer (mpi_comm, pcout,
	   dealii::TimerOutput::summary,
	   dealii::TimerOutput::wall_times)
  {
    parameters.enter_subsection ("Material");
    {
      parameters.declare_entry ("Material function", "0",
				dealii::Patterns::Anything (),
				"A functional description of the material.");

      parameters.declare_entry ("Lattice background",
				"0, 0, 0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 3, ","),
				"Size of the lattice of the background");

      parameters.declare_entry ("Lattice inclusion",
				"0, 0, 0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 3, ","),
				"Size of the lattice of an inclusion");
      
      parameters.declare_entry ("Dielectric background",
				"0, 0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 2, ","),
				"Dielectric coefficients of the background");
      
      parameters.declare_entry ("Dielectric inclusion",
				"0, 0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 2, ","),
				"Dielectric coefficients of an inclusion");

      parameters.declare_entry ("Piezoelectric background",
				"0, 0, 0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 3, ","),
				"Piezoelectric coefficients of the background");
	    
      parameters.declare_entry ("Piezoelectric inclusion",
				"0, 0, 0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 3, ","),
				"Piezoelectric coefficients of an inclusion");

      parameters.declare_entry ("Polarelectric background",
				"0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 1, ","),
				"Polarelectric coefficients of the background");
	    
      parameters.declare_entry ("Polarelectric inclusion",
				"0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 1, ","),
				"Polarelectric coefficients of an inclusion");
      
    }
    parameters.leave_subsection ();
    
    parameters.parse_input (prm);
  }

  
  /**
   * Class destructor.
   */
  template <int dim>
  PiezoelectricProblem<dim>::~PiezoelectricProblem ()
  {
    // Wipe DoF handlers.
    scalar_dof_handler.clear ();
    vector_dof_handler.clear ();
  }
  

  /**
   * Setup system matrices and vectors.
   */
  template <int dim>
  void PiezoelectricProblem<dim>::setup_system ()
  {
    dealii::TimerOutput::Scope time (timer, "setup system");

    // Under no conditions should the displacement vector be all zero.
    AssertThrow (!(*locally_relevant_displacement).all_zero (),
		 dealii::ExcMessage ("Displacement vector can not be all zero!"));
    
    // Determine locally relevant DoFs.
    scalar_dof_handler.distribute_dofs (scalar_finite_element);
    vector_dof_handler.distribute_dofs (vector_finite_element);
    
    locally_owned_dofs = scalar_dof_handler.locally_owned_dofs ();
    dealii::DoFTools::extract_locally_relevant_dofs (scalar_dof_handler, locally_relevant_dofs);

    // Initialise distributed vectors.
    locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs,
				      mpi_comm);
    system_rhs.reinit (locally_owned_dofs,
		       mpi_comm);

    // Setup hanging node constraints.
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints (scalar_dof_handler, constraints);
    dealii::DoFTools::make_zero_boundary_constraints (scalar_dof_handler, constraints);
    constraints.close ();

    // Finally, create a distributed sparsity pattern and initialise
    // the system matrix from that.
    dealii::DynamicSparsityPattern dsp (locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern (scalar_dof_handler, dsp, constraints, false);
    dealii::SparsityTools::distribute_sparsity_pattern (dsp,
							scalar_dof_handler.n_locally_owned_dofs_per_processor (),
							mpi_comm,
							locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs, locally_owned_dofs,
                          dsp, mpi_comm);

  }

  
  /**
   * Assemble the linear algebra system.
   */
  template <int dim>
  void
  PiezoelectricProblem<dim>::assemble_system ()
  {
    dealii::TimerOutput::Scope time (timer, "assemble system");
   
    // Define quadrature rule to be used.
    const dealii::QGauss<dim> quadrature_formula (3);
    
    dealii::FEValues<dim> scalar_fe_values (scalar_finite_element, quadrature_formula,
					    dealii::update_values            |
					    dealii::update_gradients         |
					    dealii::update_quadrature_points |
					    dealii::update_JxW_values);
    
    dealii::FEValues<dim> vector_fe_values (vector_finite_element, quadrature_formula,
					    dealii::update_values            |
					    dealii::update_gradients         |
					    dealii::update_quadrature_points |
					    dealii::update_JxW_values);
    
    const unsigned int dofs_per_cell = scalar_finite_element.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size ();

    dealii::FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell); 
    dealii::Vector<double> cell_vector (dofs_per_cell); 
    std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    // A vector of material values at each quadrature point and the
    // function to be parsed from the input file.
    dealii::FunctionParser<dim> material_function;

    parameters.enter_subsection ("Material");
    {
      material_function.initialize (dealii::FunctionParser<dim>::default_variable_names (),
				    parameters.get ("Material function"),
				    typename dealii::FunctionParser<dim>::ConstMap ());
    }
    parameters.leave_subsection ();

    // Displacement function values.
    std::vector<dealii::Tensor<2, dim> > displacement_function_values (n_q_points, dealii::Tensor<2, dim> ());
    
    // Material function values.
    std::vector<double> material_function_values (n_q_points);
    dealii::Tensor<2, dim> strain_function_values;   

    // Lattice coefficients from file.
    std::vector<double> lattice_coefficients_background;
    std::vector<double> lattice_coefficients_inclusion;

    // Dielectric coefficients from input file.
    std::vector<double> dielectric_coefficients_background;
    std::vector<double> dielectric_coefficients_inclusion;
    
    // Piezoelectric coefficients from input file.
    std::vector<double> piezoelectric_coefficients_background;
    std::vector<double> piezoelectric_coefficients_inclusion;

    // Polarelectric coefficients from input file.
    std::vector<double> polarelectric_coefficients_background;
    std::vector<double> polarelectric_coefficients_inclusion;

    
    parameters.enter_subsection ("Material");
    {
      lattice_coefficients_background = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Lattice background"), ','));

      lattice_coefficients_inclusion = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Lattice inclusion"), ','));

      dielectric_coefficients_background = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Dielectric background"), ','));

      dielectric_coefficients_inclusion = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Dielectric inclusion"), ','));

      piezoelectric_coefficients_background = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Piezoelectric background"), ','));

      piezoelectric_coefficients_inclusion = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Piezoelectric inclusion"), ','));

      polarelectric_coefficients_background = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Polarelectric background"), ','));

      polarelectric_coefficients_inclusion = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Polarelectric inclusion"), ','));
    }
    parameters.leave_subsection ();

    Assert (lattice_coefficients_background.size ()==lattice_coefficients_inclusion.size (),
	    dealii::ExcDimensionMismatch (lattice_coefficients_background.size (),
					  lattice_coefficients_inclusion.size ()));

    Assert (dielectric_coefficients_background.size ()==dielectric_coefficients_inclusion.size (),
	    dealii::ExcDimensionMismatch (dielectric_coefficients_background.size (),
					  dielectric_coefficients_inclusion.size ()));

    Assert (piezoelectric_coefficients_background.size ()==piezoelectric_coefficients_inclusion.size (),
	    dealii::ExcDimensionMismatch (piezoelectric_coefficients_background.size (),
					  piezoelectric_coefficients_inclusion.size ()));

    Assert (polarelectric_coefficients_background.size ()==polarelectric_coefficients_inclusion.size (),
	    dealii::ExcDimensionMismatch (polarelectric_coefficients_background.size (),
					  polarelectric_coefficients_inclusion.size ()));
    
    typename dealii::DoFHandler<dim>::active_cell_iterator
      scalar_cell = scalar_dof_handler.begin_active (),
      vector_cell = vector_dof_handler.begin_active (),
      scalar_endc = scalar_dof_handler.end ();

    for (; scalar_cell!=scalar_endc; ++scalar_cell, ++vector_cell)
      if (scalar_cell->subdomain_id () == dealii::Utilities::MPI::this_mpi_process (mpi_comm))
	{
	  cell_matrix = 0;
	  cell_vector = 0;

	  scalar_fe_values.reinit (scalar_cell);
	  vector_fe_values.reinit (vector_cell);

	  // Extract scalar- and vector-values from FEValues.
	  const dealii::FEValuesExtractors::Scalar v (0);
	  const dealii::FEValuesExtractors::Vector u (0);

	  // Obtain the material identification using scalar-values on
	  // this cell and transfer it to a strain description.
	  material_function.value_list (scalar_fe_values.get_quadrature_points (),
					material_function_values);

	  // Obtain the values of the displacements using
	  // vector-values on this cell.
	  vector_fe_values[u].get_function_gradients ((*locally_relevant_displacement),
	   					      displacement_function_values);
	  
	  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	    {

	      // Build the lattice tensor.
	      lattice_coefficients.clear ();
	      
	      for (unsigned int i=0; i<lattice_coefficients_inclusion.size (); ++i)
		lattice_coefficients.push_back (material_function_values[q_point] *
						(lattice_coefficients_inclusion[i]/lattice_coefficients_background[i]) - 1.);
	      lattice_tensor.set_coefficients (lattice_coefficients);
	      lattice_tensor.distribute_coefficients ();

	      // Build the dielectric tensor.
	      dielectric_coefficients.clear ();

	      for (unsigned int i=0; i<dielectric_coefficients_inclusion.size (); ++i)
		dielectric_coefficients.push_back (material_function_values[q_point]*dielectric_coefficients_inclusion[i] +
						   (1.-material_function_values[q_point])*dielectric_coefficients_background[i]);

	      dielectric_tensor.set_coefficients (dielectric_coefficients);
	      dielectric_tensor.distribute_coefficients ();

	      // Build the piezoelectric tensor.
	      piezoelectric_coefficients.clear ();

	      for (unsigned int i=0; i<piezoelectric_coefficients_inclusion.size (); ++i)
		piezoelectric_coefficients.push_back (material_function_values[q_point]*piezoelectric_coefficients_inclusion[i] +
						      (1.-material_function_values[q_point])*piezoelectric_coefficients_background[i]);

	      piezoelectric_tensor.set_coefficients (piezoelectric_coefficients);
	      piezoelectric_tensor.distribute_coefficients ();

	      // Build the polarelectric tensor.
	      polarelectric_coefficients.clear ();

	      for (unsigned int i=0; i<polarelectric_coefficients_inclusion.size (); ++i)
		polarelectric_coefficients.push_back (material_function_values[q_point]*polarelectric_coefficients_inclusion[i] +
						      (1.-material_function_values[q_point])*polarelectric_coefficients_background[i]);

	      polarelectric_tensor.set_coefficients (polarelectric_coefficients);
	      polarelectric_tensor.distribute_coefficients ();
	      
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  const dealii::Tensor<1, dim> v_i_grad = scalar_fe_values[v].gradient (i, q_point);
		  
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      const dealii::Tensor<1, dim> v_j_grad = scalar_fe_values[v].gradient (j, q_point);
		      
		      // Local stiffness matrix.
		      cell_matrix (i,j) +=
		       	contract (v_i_grad, dielectric_tensor, v_j_grad) *
		       	scalar_fe_values.JxW (q_point);
		      
		    } 

		  strain_function_values.clear ();
		  
		  for (unsigned int a=0; a<dim; ++a)
		    for (unsigned int b=0; b<dim; ++b)
		      strain_function_values[a][b] =
			0.5*(displacement_function_values[q_point][a][b]+displacement_function_values[q_point][a][b]);
		  
		  // Local right hand side vector.
		  cell_vector (i) +=
		    (contract (v_i_grad, piezoelectric_tensor, strain_function_values) +
		     contract (v_i_grad, polarelectric_tensor))                        *
		    scalar_fe_values.JxW (q_point);
		}
	    } // q_point
	  
	  scalar_cell->get_dof_indices (local_dof_indices);
	  
	  constraints.distribute_local_to_global (cell_matrix, cell_vector,
						  local_dof_indices,
						  system_matrix, system_rhs);
	} // cell!=endc

    system_matrix.compress (dealii::VectorOperation::add);
    system_rhs.compress (dealii::VectorOperation::add);
  }
  

  /**
   * Solve the linear algebra system.
   */
  template <int dim>
  unsigned int
  PiezoelectricProblem<dim>::solve ()
  {
    dealii::TimerOutput::Scope time (timer, "solve");
    
    dealii::PETScWrappers::MPI::Vector completely_distributed_solution (locally_owned_dofs, mpi_comm);
    dealii::SolverControl solver_control (scalar_dof_handler.n_dofs (), 1e-06);
    dealii::PETScWrappers::SolverBicgstab solver (solver_control, mpi_comm);
    dealii::PETScWrappers::PreconditionBlockJacobi preconditioner (system_matrix);
    
    solver.solve (system_matrix, completely_distributed_solution, system_rhs,
		  preconditioner);
    
    // Ensure that all ghost elements are also copied as necessary.
    constraints.distribute (completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;

    // Return the number of iterations (last step) of the solve.
    return solver_control.last_step ();
  }

  template <int dim>
  void PiezoelectricProblem<dim>::refine_grid ()
  {
    dealii::TimerOutput::Scope time (timer, "refine grid");

    // Prepare a solution transfer for vector-valued displacements.
    dealii::parallel::distributed::
      SolutionTransfer<dim, dealii::PETScWrappers::MPI::Vector> solution_transfer (vector_dof_handler);
    
    dealii::Vector<float> estimated_error_per_cell ((*triangulation).n_active_cells ());

    dealii::KellyErrorEstimator<dim>::estimate (scalar_dof_handler, dealii::QGauss<dim-1>(4),
						typename dealii::FunctionMap<dim>::type (),
						locally_relevant_solution,
						estimated_error_per_cell);
    
    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number ((*triangulation), estimated_error_per_cell,
				       0.250, 0.025);

    // Prepare grid and solution for coarsening and refinement.
    (*triangulation).prepare_coarsening_and_refinement ();
    solution_transfer.prepare_for_coarsening_and_refinement (*(locally_relevant_displacement));

    // Excecute coarsening and refinement on the grid.
    (*triangulation).execute_coarsening_and_refinement ();
    
    // Interpolate the displacement vector by redistributing
    // vector-valued degrees of freedom and actually interpolating the
    // displacement solution.
    vector_dof_handler.distribute_dofs (vector_finite_element);
    locally_owned_dofs = vector_dof_handler.locally_owned_dofs ();

    dealii::PETScWrappers::MPI::Vector interpolated_displacement (locally_owned_dofs, mpi_comm);
    solution_transfer.interpolate (interpolated_displacement);

    (*locally_relevant_displacement) = interpolated_displacement;
  }

  

  /**
   * Output results.
   */
  template <int dim>
  void
  PiezoelectricProblem<dim>::output_results (const unsigned int cycle)
  {
    dealii::TimerOutput::Scope time (timer, "output_results");

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (scalar_dof_handler);
    data_out.add_data_vector (locally_relevant_solution, "potential");

    dealii::Vector<float> subdomain ((*triangulation).n_active_cells ());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain (i) = (*triangulation).locally_owned_subdomain ();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();
    
    const std::string filename = ("potential-" +
                                  dealii::Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  dealii::Utilities::int_to_string
                                  ((*triangulation).locally_owned_subdomain (), 4));

    std::ofstream output ((filename + ".vtu").c_str ());
    data_out.write_vtu (output);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 1)
      {
	std::vector<std::string> filenames;
	
	for (unsigned int i=0;
	     i<dealii::Utilities::MPI::n_mpi_processes (mpi_comm);
	     ++i)
	  filenames.push_back ("potential-" +
			       dealii::Utilities::int_to_string (cycle, 2) +
			       "." +
			       dealii::Utilities::int_to_string (i, 4) +
			       ".vtu");
	std::ofstream master_output (("potential-" +
				      dealii::Utilities::int_to_string (cycle, 2) +
				      ".pvtu").c_str ());

	data_out.write_pvtu_record (master_output, filenames);
      }
  }
  

  /**
   * Run the application in the order specified.
   */
  template <int dim>
  void
  PiezoelectricProblem<dim>::run ()
  {
    const unsigned int n_cycles = 2;
    
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
	pcout << "PiezoelectricProblem:: Cycle " << cycle << ':'
	      << std::endl;
	
	if (cycle!=0)
	  refine_grid ();
	
	pcout << "   Number of active cells:       "
	      << (*triangulation).n_global_active_cells ()
	      << std::endl;
	
	setup_system ();
	
	pcout << "   Number of degrees of freedom: "
	      << scalar_dof_handler.n_dofs ()
	      << " + "
	      << vector_dof_handler.n_dofs ()
	      << std::endl;
	
	assemble_system ();
	
	const unsigned int n_iterations = solve ();
	
	pcout << "   Solved in " << n_iterations
	      << " iterations."
	      << std::endl;
	
	pcout << "   Linfty-norm:                  "
	      << locally_relevant_solution.linfty_norm ()
	       << std::endl;
	
	if (dealii::Utilities::MPI::n_mpi_processes (mpi_comm) <= 32)
	  output_results (cycle);
      }
  }
  
} // namespace mandy

template class mandy::PiezoelectricProblem<3>;
