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
						   MPI_Comm                                          &mpi_communicator,
						   const std::string                                 &prm)
    :
    mpi_comm (mpi_communicator),
    triangulation (&triangulation),
    dof_handler (triangulation),
    finite_element (dealii::FE_Q<dim> (2), 1),
    // finite_element (dealii::FE_Q<dim> (2), dim),
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
      
      parameters.declare_entry ("Elastic background",
				"0, 0, 0, 0, 0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 5, ","),
				"Elastic coefficients of the background");
      
      parameters.declare_entry ("Elastic inclusion",
				"0, 0, 0, 0, 0",
				dealii::Patterns::List (dealii::Patterns::Anything (), 1, 5, ","),
				"Elastic coefficients of an inclusion");

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
    dof_handler.clear ();
  }
  

  /**
   * Setup system matrices and vectors.
   */
  template <int dim>
  void PiezoelectricProblem<dim>::setup_system ()
  {
    dealii::TimerOutput::Scope time (timer, "setup system");

    // Determine locally relevant DoFs.
    dof_handler.distribute_dofs (finite_element);
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    dealii::DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);

    // Initialise distributed vectors.
    locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs,
				      mpi_comm);
    system_rhs.reinit (locally_owned_dofs,
		       mpi_comm);

    // Setup hanging node constraints.
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    dealii::DoFTools::make_zero_boundary_constraints (dof_handler, constraints);
    constraints.close ();

    // Finally, create a distributed sparsity pattern and initialise
    // the system matrix from that.
    dealii::DynamicSparsityPattern dsp (locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
    dealii::SparsityTools::distribute_sparsity_pattern (dsp,
							dof_handler.n_locally_owned_dofs_per_processor (),
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
    
    dealii::FEValues<dim> fe_values (finite_element, quadrature_formula,
				     dealii::update_values            |
				     dealii::update_gradients         |
				     dealii::update_quadrature_points |
				     dealii::update_JxW_values);
    
    const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
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

    std::vector<double> material_function_values (n_q_points);

    // Lattice coefficients from file.
    std::vector<double> lattice_coefficients_background;
    std::vector<double> lattice_coefficients_inclusion;

    // Elastic coefficients from input file.
    std::vector<double> elastic_coefficients_background;
    std::vector<double> elastic_coefficients_inclusion;

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

      elastic_coefficients_background = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Elastic background"), ','));

      elastic_coefficients_inclusion = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Elastic inclusion"), ','));

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
    
    Assert (elastic_coefficients_background.size ()==elastic_coefficients_inclusion.size (),
	    dealii::ExcDimensionMismatch (elastic_coefficients_background.size (),
					  elastic_coefficients_inclusion.size ()));

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
      cell = dof_handler.begin_active (),
      endc = dof_handler.end ();
    
    for (; cell!=endc; ++cell)
      if (cell->subdomain_id () == dealii::Utilities::MPI::this_mpi_process (mpi_comm))
	{
	  cell_matrix = 0;
	  cell_vector = 0;
	  fe_values.reinit (cell);

	  // Extract vector-values from FEValues.
	  const dealii::FEValuesExtractors::Vector u (0);

	  // Obtain the material identification on this cell and
	  // transfer it to a strain description.
	  material_function.value_list (fe_values.get_quadrature_points (),
					material_function_values);
  
	  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	    {

	      // Build the lattice tensor.
	      lattice_coefficients.clear ();
	      
	      for (unsigned int i=0; i<lattice_coefficients_inclusion.size (); ++i)
		lattice_coefficients.push_back (material_function_values[q_point] *
						(lattice_coefficients_inclusion[i]/lattice_coefficients_background[i]) - 1.);
	      lattice_tensor.set_coefficients (lattice_coefficients);
	      lattice_tensor.distribute_coefficients ();

	      // Build the elastic tensor.
	      elastic_coefficients.clear ();

	      for (unsigned int i=0; i<elastic_coefficients_inclusion.size (); ++i)
		elastic_coefficients.push_back (material_function_values[q_point]*elastic_coefficients_inclusion[i] +
						(1.-material_function_values[q_point])*elastic_coefficients_background[i]);

	      elastic_tensor.set_coefficients (elastic_coefficients);
	      elastic_tensor.distribute_coefficients ();

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
		  // const dealii::Tensor<2, dim> u_i_grad = fe_values[u].symmetric_gradient (i, q_point);
		  
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      // const dealii::Tensor<2, dim> u_j_grad = fe_values[u].symmetric_gradient (j, q_point);
		      
		      // Local stiffness matrix.
		      // cell_matrix (i,j) +=
		      // 	contract (u_i_grad, elastic_tensor, u_j_grad) *
		      // 	fe_values.JxW (q_point);
		      
		    } 
		  
		  // Local right hand side vector.
		  // cell_vector (i) +=
		  //   contract (u_i_grad, elastic_tensor, lattice_tensor) *
		  //   fe_values.JxW (q_point);
		  
		}

	    } // q_point
	  
	  cell->get_dof_indices (local_dof_indices);
	  
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
    dealii::SolverControl solver_control (dof_handler.n_dofs (), 1e-06);
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


  /**
   * Output results.
   */
  template <int dim>
  void
  PiezoelectricProblem<dim>::output_results (const unsigned int cycle)
  {
    dealii::TimerOutput::Scope time (timer, "output_results");

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (locally_relevant_solution, "displacement");

    dealii::Vector<float> subdomain ((*triangulation).n_active_cells ());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain (i) = (*triangulation).locally_owned_subdomain ();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();
    
    const std::string filename = ("displacement-" +
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
	  filenames.push_back ("displacement-" +
			       dealii::Utilities::int_to_string (cycle, 2) +
			       "." +
			       dealii::Utilities::int_to_string (i, 4) +
			       ".vtu");
	std::ofstream master_output (("displacement-" +
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
    const unsigned int n_cycles = 1;
    
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
	
	pcout << "PiezoelectricProblem:: Cycle " << cycle << ':'
	      << std::endl;
	
	pcout << "   Number of active cells:       "
	      << (*triangulation).n_global_active_cells ()
	      << std::endl;
	
	setup_system ();
	
	pcout << "   Number of degrees of freedom: "
	      << dof_handler.n_dofs ()
	      << std::endl;
	
	assemble_system ();
	
	// const unsigned int n_iterations = solve ();
	
	// pcout << "   Solved in " << n_iterations
	//       << " iterations."
	//       << std::endl;
	
	// pcout << "   Linfty-norm:                  "
	//       << locally_relevant_solution.linfty_norm ()
	//       << std::endl;

	// if (dealii::Utilities::MPI::n_mpi_processes (mpi_comm) <= 32)
	//   output_results (cycle);
      }
  }
  
} // namespace mandy

template class mandy::PiezoelectricProblem<3>;
