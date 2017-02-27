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

#include <iostream>

#ifndef __mandy_tensor_base_h
#define __mandy_tensor_base_h

namespace mandy
{
  
  /**
   * A base class that describes the tensors of coefficients.
   */ 
  template <int rank, int dim, typename Value = double>
    class TensorBase
    :
    public dealii::Tensor<rank, dim, Value>
    {
    public:
    
    /**
     * Constructor.
     */
    TensorBase ()
    :
    tensor (dealii::Tensor<rank, dim, Value> ())
    {};

    /**
     * Distribute @p coefficients.
     */
    virtual void distribute_coefficients ();

    /**
     *
     */
    virtual bool is_symmetric (const Value tolerance);
    
    /**
     * Set @p coefficients.
     */
    void set_coefficients (std::vector<Value> &coefficients);

    /**
     * Print to screen.
     */
    void print ();

    /**
     * Read only access to the underlying data type.
     */
    inline
    dealii::Tensor<rank, dim, Value> operator* () const
    {
      return this->tensor;
    }
    
    protected:
    
    /**
     * The underlying data type that describes an elastic tensor.
     */
    dealii::Tensor<rank, dim, Value> tensor;
    
    /**
     * A vector of coefficients.
     */
    std::vector<Value> coefficients_;
    
    }; // TensorBase

  /**
   * Contract all indices in the set of tensors. 
   *
   * @note This is a specialisation of deal.II's contract3 function.
   */
  template <int rank, int dim, typename Value>
  inline
    Value contract (const dealii::Tensor<rank, dim, Value>    &src1,
		    const mandy::TensorBase<rank, dim, Value> &src2)
  {
    return dealii::scalar_product (src1, *src2);
  }
  
  /**
   * Contract all indices in the set of tensors. 
   *
   * @note This is a specialisation of deal.II's contract3 function.
   */
  template <int rank_1, int rank_2, int dim, typename Value>
  inline
    Value contract (const dealii::Tensor<rank_1, dim, Value>           &src1,
		    const mandy::TensorBase<rank_1+rank_2, dim, Value> &src2,
		    const dealii::Tensor<rank_2, dim, Value>           &src3)
  {
    return dealii::contract3 (src1, *src2, src3);
  }

  /**
   * Contract all indices in the set of tensors. 
   *
   * @note This is a specialisation of deal.II's contract3 function.
   */
  template <int rank_1, int rank_2, int dim, typename Value>
    inline
    Value contract (const dealii::Tensor<rank_1, dim, Value>           &src1,
		    const mandy::TensorBase<rank_1+rank_2, dim, Value> &src2,
		    const mandy::TensorBase<rank_2, dim, Value>        &src3)
  {
    return dealii::contract3 (src1, *src2, *src3);
  }
  
} // namepsace mandy

#endif // __mandy_tensor_base_h
