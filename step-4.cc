
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

#include <mandy/elastic_tensor.h>
#include <mandy/matrix_creator.h>
#include <mandy/vector_creator.h>

#include <mandy/crystal_symmetry_group.h>

#include <fstream>
#include <iostream>

#include <algorithm>    // std::transform
#include <functional>   // std::plus

namespace mandy
{

  /**
   * Solve the system Mx=b, where M is the mass matrix and b is a
   * linear function.
   */
  template <int dim>
  class MaterialID
  {
  public:

    /**
     * Class constructor.
     */
    MaterialID (const std::string &prm);

    /**
     * Class destructor.
     */
    ~MaterialID ();

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
     * Assemble system matrices and vectors.
     */
    void assemble_system();

    /**
     * Solve the linear algebra system.
     */
    unsigned int solve ();
    
    /**
     * Output results, ie., finite element functions and derived
     * quantitites for this cycle.
     */
    void output_results (const unsigned int cycle);

    /**
     * Refine grid based on Kelly's error estimator working on the
     * material id (solution vector).
     */
    void refine_grid ();
    
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

    /**
     * Scalar valued finite element primarily used for interpolating
     * material iudentification.
     */
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
    
  }; // MaterialID

  
  /**
   * Class constructor.
   */
  template <int dim>
  MaterialID<dim>::MaterialID (const std::string &prm)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename dealii::Triangulation<dim>::MeshSmoothing
                   (dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening)),
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

    parameters.declare_entry ("MaterialID", "0",
                              dealii::Patterns::Anything (),
                              "A functional description of the material ID.");
    
    parameters.parse_input (prm);
  }

  
  /**
   * Class destructor.
   */
  template <int dim>
  MaterialID<dim>::~MaterialID ()
  {
    // Wipe DoF handlers.
    dof_handler.clear ();
  }


  /**
   * Make initial coarse grid.
   */
  template <int dim>
  void
  MaterialID<dim>::make_coarse_grid ()
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
  void MaterialID<dim>::setup_system ()
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
  MaterialID<dim>::assemble_system ()
  {
    dealii::TimerOutput::Scope time (timer, "assemble system");

    // Define quadrature rule to be used.
    const dealii::QGauss<dim> quadrature_formula (3);

    // Initialise the function parser.
    dealii::FunctionParser<dim> material_identification;
    material_identification.initialize (dealii::FunctionParser<dim>::default_variable_names (),
					parameters.get ("MaterialID"),
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
  MaterialID<dim>::solve ()
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
  MaterialID<dim>::output_results (const unsigned int cycle)
  {
    dealii::TimerOutput::Scope time (timer, "output_results");

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (locally_relevant_solution, "material_id");

    dealii::Vector<float> subdomain (triangulation.n_active_cells ());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain (i) = triangulation.locally_owned_subdomain ();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();
    
    const std::string filename = ("material_id-" +
                                  dealii::Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  dealii::Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain (), 4));

    std::ofstream output ((filename + ".vtu").c_str ());
    data_out.write_vtu (output);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 1)
      {
	std::vector<std::string> filenames;
	
	for (unsigned int i=0;
	     i<dealii::Utilities::MPI::n_mpi_processes (mpi_communicator);
	     ++i)
	  filenames.push_back ("material_id-" +
			       dealii::Utilities::int_to_string (cycle, 2) +
			       "." +
			       dealii::Utilities::int_to_string (i, 4) +
			       ".vtu");
	std::ofstream master_output (("material_id-" +
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
  void MaterialID<dim>::refine_grid ()
  {
    dealii::TimerOutput::Scope time (timer, "refine grid");

    dealii::Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    
    dealii::KellyErrorEstimator<dim>::estimate (dof_handler, dealii::QGauss<dim-1>(4),
						typename dealii::FunctionMap<dim>::type (),
						locally_relevant_solution,
						estimated_error_per_cell);

    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number (triangulation,
				       estimated_error_per_cell,
				       0.250, 0.025);

    triangulation.execute_coarsening_and_refinement ();
  }
  
  
  /**
   * Run the application in the order specified.
   */
  template <int dim>
  void
  MaterialID<dim>::run ()
  {
    const unsigned int n_cycles = 5;
    
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        pcout << "MaterialID:: Cycle " << cycle << ':'
	      << std::endl;

	if (cycle==0)
	  make_coarse_grid ();

	else
	  refine_grid ();
	
	pcout << "   Number of active cells:       "
	      << triangulation.n_global_active_cells ()
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

namespace mandy
{
  /**
   * Solve the system Ax=b, where A is the vector-valued operator of
   * the Laplace-type and b is a linear function.
   */
  template <int dim>
  class ElasticProblem
  {
  public:

    /**
     * Class constructor.
     */
    ElasticProblem (const std::string &prm);

    /**
     * Class destructor.
     */
    ~ElasticProblem ();

    /**
     * Wrapper function, that controls the order of excecution.
     */
    void run ();

    /**
     * Get coefficients from the parameter file specified at run time.
     */
    void get_coefficients ();
    
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
     * Assemble system matrices and vectors.
     */
    void assemble_system();

    /**
     * Solve the linear algebra system.
     */
    unsigned int solve ();
    
    /**
     * Output results, ie., finite element functions and derived
     * quantitites for this cycle.
     */
    void output_results (const unsigned int cycle);

    /**
     * Refine grid based on Kelly's error estimator working on the
     * material id (solution vector).
     */
    void refine_grid ();
    
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

    /**
     * Scalar valued finite element primarily used for interpolating
     * material iudentification.
     */
    dealii::FESystem<dim> finite_element;

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
    
    /**
     * Tensor of elastic coefficients.
     */
    mandy::Physics::ElasticTensor<mandy::CrystalSymmetryGroup::wurtzite> elastic_tensor;

    /**
     * Vector of elastic coefficients.
     */
    std::vector<double> elastic_coefficients;
    
  }; // LinearElasticity


  /**
   * Class constructor.
   */
  template <int dim>
  ElasticProblem<dim>::ElasticProblem (const std::string &prm)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename dealii::Triangulation<dim>::MeshSmoothing
                   (dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening)),
    dof_handler (triangulation),
    finite_element (dealii::FE_Q<dim> (2), dim),
    // ---
    pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process (mpi_communicator) == 0)),
    timer (mpi_communicator, pcout,
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
    }
    parameters.leave_subsection ();
    
    parameters.parse_input (prm);
  }

  
  /**
   * Class destructor.
   */
  template <int dim>
  ElasticProblem<dim>::~ElasticProblem ()
  {
    // Wipe DoF handlers.
    dof_handler.clear ();
  }
  

  /**
   * Setup system matrices and vectors.
   */
  template <int dim>
  void ElasticProblem<dim>::setup_system ()
  {
    dealii::TimerOutput::Scope time (timer, "setup system");

    // Determine locally relevant DoFs.
    dof_handler.distribute_dofs (finite_element);
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
   * Assemble the linear algebra system.
   */
  template <int dim>
  void
  ElasticProblem<dim>::assemble_system ()
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

    // Get lattice parameters from file.
    dealii::Tensor<2, dim> lattice_parameters;
    
    // Get elastic coefficients from input file.
    std::vector<double> elastic_coefficients_background;
    std::vector<double> elastic_coefficients_inclusion;
        
    parameters.enter_subsection ("Material");
    {
      elastic_coefficients_background = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Elastic background"), ','));

      elastic_coefficients_inclusion = dealii::Utilities::string_to_double
	(dealii::Utilities::split_string_list (parameters.get ("Elastic inclusion"), ','));
    }
    parameters.leave_subsection ();
    
    AssertThrow (elastic_coefficients_background.size ()==elastic_coefficients_inclusion.size (),
		 dealii::ExcDimensionMismatch (elastic_coefficients_background.size (),
					       elastic_coefficients_inclusion.size ()));
   
    typename dealii::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active (),
      endc = dof_handler.end ();
    
    for (; cell!=endc; ++cell)
      if (cell->subdomain_id () == dealii::Utilities::MPI::this_mpi_process (mpi_communicator))
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
	          
	      elastic_coefficients.clear ();

	      for (unsigned int i=0; i<elastic_coefficients_inclusion.size (); ++i)
		elastic_coefficients.push_back (material_function_values[q_point]*elastic_coefficients_inclusion[i] +
						(1.-material_function_values[q_point])*elastic_coefficients_background[i]);

	      elastic_tensor.set_coefficients (elastic_coefficients);
	      elastic_tensor.distribute_coefficients ();
	      
	      const double distribution_ratio = 0.01; // Hard-coded (inclusion-reference)/reference ratio
	      
	      for (unsigned int a=0; a<dim; ++a)
		for (unsigned int b=0; b<dim; ++b)
		  (a==b)
		    ? lattice_parameters[a][b] = material_function_values[q_point] * distribution_ratio - 1.
		    : 0;

		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		    {
		      const dealii::Tensor<2, dim> u_i_grad = fe_values[u].symmetric_gradient (i, q_point);
		      
		      for (unsigned int j=0; j<dofs_per_cell; ++j)
			{
			  const dealii::Tensor<2, dim> u_j_grad = fe_values[u].symmetric_gradient (j, q_point);
			  
			  // Local stiffness matrix.
			  cell_matrix (i,j) +=
			    contract (u_i_grad, elastic_tensor, u_j_grad) *
			    fe_values.JxW (q_point);
			  
			} 
			  
		      // Local right hand side vector.
		      cell_vector (i) +=
			contract (u_i_grad, elastic_tensor, lattice_parameters) *
			fe_values.JxW (q_point);
		      
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
  ElasticProblem<dim>::solve ()
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
   * Run the application in the order specified.
   */
  template <int dim>
  void
  ElasticProblem<dim>::run ()
  {
    // To solve the Elastic problem, first we need to solve the
    // problem of a mterial identification. This is done by solving a
    // function against the mass matrix.

    // Once the solution vector from the above problem has been
    // computed, the elastic problem can be solved.
    



    // Create a coarse grid according to the parameters given in the
    // input file.
    dealii::GridGenerator::hyper_cube (triangulation, -5, 5);
    triangulation.refine_global (3);
    
    pcout << "   Number of active cells:       "
	  << triangulation.n_global_active_cells ()
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
      mandy::ElasticProblem<3> elastic_problem ("elastic.prm");
      elastic_problem.run ();

      // mandy::MaterialID<3> material_id ("step-4.prm");
      // material_id.run ();
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
