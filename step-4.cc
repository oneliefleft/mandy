


#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>

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

    void run ();
    
  private:

    /**
     * Input parameter file.
     */
    dealii::ParameterHandler parameters;
    
  };

  /**
   * Class constructor.
   */
  template <int dim>
  LinearElasticity<dim>::LinearElasticity (const std::string &prm_file)
  // :
  // dof_handler (triangulation),
  // fe (dealii::FE_Q<dim>(2), dim)
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
    // dof_handler.clear ();
  }
  
  template <int dim>
  void LinearElasticity<dim>::run ()
  {
  }
  
} // namespace ewalena




int main ()
{
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
