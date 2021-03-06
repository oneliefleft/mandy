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

#ifndef __mandy_vector_creator_h
#define __mandy_vector_creator_h

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
    template<int dim, int spacedim = dim, typename Value = double>
      void
      create_right_hand_side_vector (const dealii::FiniteElement<dim,spacedim> &finite_element,
				     const dealii::DoFHandler<dim,spacedim>    &dof_handler,
				     const dealii::Quadrature<dim>             &quadrature,
				     dealii::PETScWrappers::MPI::Vector        &vector,
				     const dealii::ConstraintMatrix            &constraints,
				     const dealii::FunctionParser<dim>         &function_parser,
				     const MPI_Comm                            &mpi_communicator);
    
  } // namespace VectorCreator
  
} // namepsace mandy

#endif //  __mandy_vector_creator_h
