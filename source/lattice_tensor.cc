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

#include <mandy/lattice_tensor.h>

namespace mandy
{

  namespace Physics
  {

    template<enum CrystalSymmetry crystal_symmetry, typename Value>
    void
    LatticeTensor<crystal_symmetry, Value>::distribute_coefficients ()
    {
      switch (crystal_symmetry)
	{
	case wurtzite:
	  this->distribute_coefficients_wurtzite ();
	  break;

	// case zincblende:
	//   this->distribute_coefficients_zincblende ();
	//   break;

	default:
	  dealii::ExcNotImplemented ();
	}
    }

    template<enum CrystalSymmetry crystal_symmetry, typename Value>
    void
    LatticeTensor<crystal_symmetry, Value>::distribute_coefficients_wurtzite ()
    {
      // There should be five independent coefficients.
      AssertThrow (this->coefficients_.size ()==2,
		   dealii::ExcDimensionMismatch (this->coefficients_.size (), 2));
      
      // Distribute the coefficients on to the tensor. It seems
      // there is no automagic way to do this, so just insert those
      // elements that are non-zero: L_11 = L_22, L_33.
      this->tensor = 0;
      
      // L_11 = L_22 \mapsto
      this->tensor[0][0] = this->coefficients_[0];
      this->tensor[1][1] = this->coefficients_[0];
      
      // L_33 \mapsto
      this->tensor[2][2] = this->coefficients_[1];
    }
    
  } // namespace Physics

} // namepsace mandy

template class
mandy::Physics::LatticeTensor<mandy::Physics::CrystalSymmetry::wurtzite, double>;
