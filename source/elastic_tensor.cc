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

    template<enum CrystalSymmetryGroup CSG, typename ValueType>
    bool
    ElasticTensor<CSG, ValueType>::is_symmetric (const ValueType tolerance)
    {
      bool is_symmetric = true;
      
      for (unsigned int i=0; i<3; ++i)
	for (unsigned int j=0; j<3; ++j)
	  for (unsigned int k=0; k<3; ++k)
	    for (unsigned int l=0; l<3; ++l)
	      {
		if ((std::fabs (this->tensor[i][j][k][l]-this->tensor[i][j][k][l]) > tolerance) ||
		    (std::fabs (this->tensor[i][j][k][l]-this->tensor[i][j][l][k]) > tolerance) ||
		    (std::fabs (this->tensor[i][j][k][l]-this->tensor[j][i][k][l]) > tolerance) ||
		    (std::fabs (this->tensor[i][j][k][l]-this->tensor[l][k][j][i]) > tolerance))
		  {
		    is_symmetric = false;
		    break;
		  }
	      }
      
      return is_symmetric;
    }
    
    template<enum CrystalSymmetryGroup CSG, typename ValueType>
    void
    ElasticTensor<CSG, ValueType>::distribute_coefficients ()
    {
      // There should be five independent coefficients.
      AssertThrow (this->coefficients_.size ()==5,
		   dealii::ExcDimensionMismatch (this->coefficients_.size (), 5));
      
      // Distribute the coefficients on to the tensor. It seems
      // there is no automagic way to do this, so just insert those
      // elements that are non-zero: C_11 = C_22, C_12, C_13 = C_23,
      // C_33, C_44 = C_55. 
      this->tensor = 0;
      
      // C_11 = C_22 \mapsto
      this->tensor[0][0][0][0] = this->coefficients_[0];
      this->tensor[1][1][1][1] = this->coefficients_[0];
      
      // C_12 \mapsto
      this->tensor[0][0][1][1] = this->coefficients_[1];
      this->tensor[1][1][0][0] = this->coefficients_[1];
      
      // C_13 = C_23 \mapsto
      this->tensor[0][0][2][2] = this->coefficients_[2];
      this->tensor[2][2][0][0] = this->coefficients_[2];
      this->tensor[1][1][2][2] = this->coefficients_[2];
      this->tensor[2][2][1][1] = this->coefficients_[2];
      
      // C_33 \mapsto
      this->tensor[2][2][2][2] = this->coefficients_[3];
      
      // C_44 = C55 \mapsto
      this->tensor[1][2][1][2] = this->coefficients_[4];
      this->tensor[2][1][1][2] = this->coefficients_[4];
      this->tensor[2][1][2][1] = this->coefficients_[4];
      this->tensor[1][2][2][1] = this->coefficients_[4];
      
      this->tensor[2][0][2][0] = this->coefficients_[4];
      this->tensor[0][2][2][0] = this->coefficients_[4];
      this->tensor[0][2][0][2] = this->coefficients_[4];
      this->tensor[2][0][0][2] = this->coefficients_[4];
      
      // C_66 \mapsto
      const double coefficient = (this->coefficients_[0] - this->coefficients_[1]) /2.;
      this->tensor[0][1][0][1] = coefficient;
      this->tensor[1][0][0][1] = coefficient;
      this->tensor[1][0][1][0] = coefficient;
      this->tensor[0][1][1][0] = coefficient;	
    }
  } // namespace Physics

} // namepsace mandy

template class
mandy::Physics::ElasticTensor<mandy::CrystalSymmetryGroup::wurtzite, double>;
