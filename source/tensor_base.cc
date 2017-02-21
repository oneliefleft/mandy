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

#include <mandy/tensor_base.h>

namespace mandy
{

  template<int rank, int dim, typename ValueType>
  void
  TensorBase<rank, dim, ValueType>::distribute_coefficients ()
  {
    AssertThrow (false, dealii::ExcPureFunctionCalled ());
  }
  
  template<int rank, int dim, typename ValueType>
  bool
  TensorBase<rank, dim, ValueType>::is_symmetric (const ValueType /*tolerance*/)
  {
    AssertThrow (false, dealii::ExcPureFunctionCalled ());
  }
  
  template<int rank, int dim, typename ValueType>
  void
  TensorBase<rank, dim, ValueType>::set_coefficients (std::vector<ValueType> &coefficients)
  {
    coefficients_.clear ();
    
    for (unsigned int i=0; i<coefficients.size (); ++i)
      coefficients_.push_back (coefficients[i]);
  }
  
  template<int rank, int dim, typename ValueType>
  void
  TensorBase<rank, dim, ValueType>::print ()
  {
    std::cout << this->tensor;
  }
  
} // namepsace mandy

template class mandy::TensorBase<2, 3, double>;
template class mandy::TensorBase<3, 3, double>;
template class mandy::TensorBase<4, 3, double>;
