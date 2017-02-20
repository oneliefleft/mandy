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

#include <mandy/crystal_symmetry_group.h>
#include <mandy/tensor_base.h>

#ifndef __mandy_elastic_tensor_h
#define __mandy_elastic_tensor_h

namespace mandy
{
  
  namespace Physics
  {
    
    /**
     * A class that describes the elastic tensor (or stress-strain
     * tensor).
     */ 
    template <enum CrystalSymmetryGroup, typename ValueType = double>
      class ElasticTensor
      :
      public mandy::TensorBase<4,3,ValueType>
      {
      public:
      
      /**
       * Constructor.
       */
      ElasticTensor () {};
      
      /**
       * Distribute @p coefficients
       */ 
      void distribute_coefficients ();

      /**
       * Explicitly set symmetry of this tensor.
       */ 
      bool is_symmetric (const ValueType tolerance = 1e-09);
      
    }; // ElasticTensor
    
  } // namespace Physics
  
} // namepsace mandy

#endif // __mandy_elastic_tensor_h
