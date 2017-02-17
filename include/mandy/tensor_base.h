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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>

#ifndef __mandy_tensor_base_h
#define __mandy_tensor_base_h

namespace mandy
{
  
  /**
   * A base class that describes the tensors of coefficients.
   */ 
  template <int rank, int dim, typename ValueType = double>
    class TensorBase
    :
    dealii::Tensor<rank, dim, ValueType>
    {
    public:
    
    /**
     * Constructor.
     */
    TensorBase ()
    :
    tensor (dealii::Tensor<rank, dim, ValueType> ())
    {};

    /**
     * Distribute @p coefficients.
     */
    virtual void distribute_coefficients ();

    /**
     * Set @p coefficients.
     */
    void set_coefficients (std::vector<ValueType> &coefficients);

    /**
     * Print to screen.
     */
    void print ();
    
    protected:
    
    /**
     * The underlying data type that describes an elastic tensor.
     */
    dealii::Tensor<rank, dim, ValueType> tensor;
    
    /**
     * A vector of coefficients.
     */
    std::vector<ValueType> coefficients_;
    
    }; // TensorBase

} // namepsace mandy

#endif // __mandy_tensor_base_h
