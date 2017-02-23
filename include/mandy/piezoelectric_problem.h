
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

#include <mandy/dielectric_tensor.h>
#include <mandy/elastic_tensor.h>
#include <mandy/lattice_tensor.h>
#include <mandy/piezoelectric_tensor.h>
#include <mandy/polarelectric_tensor.h>

#include <mandy/crystal_symmetry_group.h>

#include <fstream>
#include <iostream>

#include <algorithm>    // std::transform
#include <functional>   // std::plus

#ifndef __piezoelectric_problem_h
#define __piezoelectric_problem_h

namespace mandy
{
  /**
   * Solve the system Ax=b, where A is the vector-valued operator of
   * the Laplace-type and b is a linear function.
   */
  template <int dim>
  class PiezoelectricProblem
  {
  public:

    /**
     * Class constructor.
     */
    PiezoelectricProblem (dealii::parallel::distributed::Triangulation<dim> &triangulation,
			  dealii::PETScWrappers::MPI::Vector                &locally_relevant_displacement,
			  MPI_Comm                                          &mpi_communicator,
			  const std::string                                 &prm);

    /**
     * Class destructor.
     */
    ~PiezoelectricProblem ();

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
    MPI_Comm mpi_comm;

    /**
     * A distributed grid on which all computations are done.
     */
    const dealii::SmartPointer<dealii::parallel::distributed::Triangulation<dim> > triangulation;

    /**
     * Scalar DoF handler primarily used for interpolating material
     * identification.
     */
    dealii::DoFHandler<dim> dof_handler;

    /**
     * Scalar DoF handler primarily used for interpolating material
     * identification.
     */
    dealii::DoFHandler<dim> scalar_dof_handler;

    /**
     * Scalar DoF handler primarily used for interpolating material
     * identification.
     */
    dealii::DoFHandler<dim> vector_dof_handler;

    /**
     * Scalar valued finite element.
     */
    dealii::FESystem<dim> scalar_finite_element;

    /**
     * Vector valued finite element.
     */
    dealii::FESystem<dim> vector_finite_element;

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
     * Locally relevant displacement vector.
     */
    const dealii::SmartPointer<dealii::PETScWrappers::MPI::Vector> locally_relevant_displacement;
    
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
     * Tensor of dielectric coefficients.
     */
    mandy::Physics::DielectricTensor<mandy::CrystalSymmetryGroup::wurtzite> dielectric_tensor;

    /**
     * Vector of dielectric coefficients.
     */
    std::vector<double> dielectric_coefficients;

    /**
     * Tensor of piezoelectric coefficients.
     */
    mandy::Physics::PiezoelectricTensor<mandy::CrystalSymmetryGroup::wurtzite> piezoelectric_tensor;

    /**
     * Vector of piezoelectric coefficients.
     */
    std::vector<double> piezoelectric_coefficients;

    /**
     * Tensor of polarelectric coefficients.
     */
    mandy::Physics::PolarelectricTensor<mandy::CrystalSymmetryGroup::wurtzite> polarelectric_tensor;

    /**
     * Vector of polarelectric coefficients.
     */
    std::vector<double> polarelectric_coefficients;

    /**
     * Tensor of lattice coefficients.
     */
    mandy::Physics::LatticeTensor<mandy::CrystalSymmetryGroup::wurtzite> lattice_tensor;

    /**
     * Vector of lattice coefficients.
     */
    std::vector<double> lattice_coefficients;

  }; // PiezoelectricProblem
  
} // namespace mandy

#endif // __piezoelectric_problem_h
