// -----------------------------------------------------------------------------
// 
// BSD 2-Clause License
// 
// Copyright (c) 2017, mandy authors
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// -----------------------------------------------------------------------------

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>

namespace mandy
{

  namespace VectorCreator
  {

    /**
     * Assemble the right hand side function. If no coefficient is
     * given (i.e., if the pointer to a function object is zero as it
     * is by default), the coefficient is taken as being constant and
     * equal to one.
     *
     * The argument @p constraints allows to apply constraints on the
     * resulting matrix directly.
     */
    template<int dim, int spacedim = dim, typename number = double>
      void
      create_right_hand_side_vector (const dealii::FESystem<dim,spacedim>     &finite_element,
				     const dealii::DoFHandler<dim,spacedim>   &dof_handler,
				     const dealii::Quadrature<dim>            &quadrature,
				     dealii::PETScWrappers::MPI::Vector       &vector,
				     const dealii::ConstraintMatrix           &constraints,
				     const dealii::FunctionParser<dim>        &function_parser,
				     const MPI_Comm                           &mpi_communicator)
      {
	dealii::FEValues<dim> fe_values (finite_element, quadrature,
					 dealii::update_values            |
					 dealii::update_quadrature_points |
					 dealii::update_JxW_values);
	
	const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
	const unsigned int n_q_points    = quadrature.size ();
	
	dealii::Vector<double> cell_vector (dofs_per_cell); 
	std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);

	std::vector<double> function_values (n_q_points);
	
	typename dealii::DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active (),
	endc = dof_handler.end ();
	
	for (; cell!=endc; ++cell)
	  if (cell->subdomain_id () == dealii::Utilities::MPI::this_mpi_process (mpi_communicator))
	    {
	      cell_vector = 0;
	      fe_values.reinit (cell);

	      function_parser.value_list (fe_values.get_quadrature_points (),
					  function_values);
	      
	      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		for (unsigned int j=0; j<dofs_per_cell; ++j)
		   {
		     // Local right hand side vector.
		     cell_vector (j) +=
		       function_values[q_point]          * 
		       fe_values.shape_value (j,q_point) *
		       fe_values.JxW (q_point);
		   }
	      
	      cell->get_dof_indices (local_dof_indices);
	      
	     constraints.distribute_local_to_global (cell_vector,
						     local_dof_indices,
						     vector);
	    } // cell!=endc
	
	vector.compress (dealii::VectorOperation::add);
      }
    
  } // namespace VectorCreator
  
} // namepsace mandy
