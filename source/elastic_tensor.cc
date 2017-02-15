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

#include <mandy/elastic_tensor.h>

namespace mandy
{

  namespace Physics
  {

    template<typename number>
    void
    ElasticTensor<number>::distribute_coefficients ()
    {
      // There should be five independent coefficients.
      AssertThrow (coefficients.size ()==5,
		   dealii::ExcDimensionMismatch (coefficients.size (), 4));

      // Distribute the coefficients on to the tensor. It seems
      // there is no automagic way to do this, so just insert those
      // elements that are non-zero: C_11 = C_22, C_12, C_13 = C_23,
      // C_33, C_44 = C_55. 
      tensor = 0;
      
      // C_11 = C_22 \mapsto
      tensor[0][0][0][0] = tensor[1][1][1][1] = coefficients[0];
      
      // C_12 \mapsto
      tensor[0][0][1][1] = tensor[1][1][0][0] = coefficients[1];
      
      // C_13 = C_23 \mapsto
      tensor[0][0][2][2] = tensor[1][1][2][2] = coefficients[2];
      
      // C_33 \mapsto
      tensor[2][2][2][2] = coefficients[3];
      
      // C_44 = C55 \mapsto
      tensor[1][2][1][2] = tensor[2][1][1][2] = tensor[2][1][2][1] = tensor[1][2][2][1]
	=
	tensor[2][0][2][0] = tensor[0][2][2][0] = tensor[0][2][0][2] = tensor[2][0][0][2]
	=
	coefficients[4];
      
      // C_66 \mapsto
      tensor[0][1][0][1] = tensor[1][0][0][1] = tensor[1][0][1][0] = tensor[0][1][1][0]
	=
	(coefficients[0] - coefficients[1]) /2.;
      
    }
    
  } // namespace Physics

} // namepsace mandy

template class
mandy::Physics::ElasticTensor<double>;
