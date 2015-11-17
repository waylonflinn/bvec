# internal imports
from bvec import carray_ext

import bvec

# external imports
import numpy as np
import bcolz

class carray(bcolz.carray):

	def divide(self, denominator, out=None):
		'''
			Divide a carray by an input carray, elementwise.

		Arguments:
			array (carray): 1-dimensional array to divide this carray by.

			out: container to be used to hold the result (numpy.ndarray or bcolz.carray)
		'''

		out = self.empty_like(shape=(self.shape[0],))

		if(isinstance(denominator, bcolz.carray)):
			carray_ext._divide_carray(self, denominator, denominator[0], out)
		else:
			# scalar division
			carray_ext._divide_scalar(self, denominator, out)

		return out


	def dot(self, matrix, out=None):
		'''
			Dot product of two arrays, with bcolz.carray support. The `out`
			parameter may be used to specify a carray container. If no `out`
			parameter is specified, a numpy.ndarray is created.

			If `out` is provided it's shape must match the output which would be
			constructed, or an error will be raised. If you want to use the `out`
			parameter, but aren't sure how, use `bvec.carray.empty_like_dot()`

		Arguments:
			matrix (carray): two dimensional matrix in a bcolz.carray, row
			vector format or a one dimensional numpy.ndarray.

			out: container to be used to hold the result (numpy.ndarray or bcolz.carray)
		'''


		# check dtype compatibility
		if self.dtype.type != matrix.dtype.type:
			raise ValueError("inputs must have the same dtype. Found {0} and {1}".format(self.dtype, matrix.dtype))

		# check shape
		if( (self.shape[1] != matrix.shape[0]) and
			(type(matrix) == carray) and (self.shape[1] != matrix.shape[1])):
			raise ValueError("inputs must have compatible shapes. Found {0} and {1}".format(self.shape, matrix.shape))

		# create output container, if none is supplied
		if out is None:
			out = self.empty_like_dot(matrix)

		if type(matrix) == np.ndarray:
			# input is a vector

			assert len(matrix.shape) == 1

			# check output container
			assert len(out.shape) == 1

			if type(out) == np.ndarray:
				assert out.shape[0] == self.shape[0]
				carray_ext._dot(self, matrix, out)

				return out
			else:
				# output carray
				assert isinstance(out, bcolz.carray)

				carray_ext._dot_carray(self, matrix, out)
				return out
		else:
			# input is a matrix

			assert len(matrix.shape) == 2

			# check output container
			assert len(out.shape) == 2
			assert out.shape[1] == matrix.shape[0]


			if type(out) == np.ndarray:
				carray_ext._dot_mat(self, matrix, out)
			else:
				# output carray
				assert isinstance(out, bcolz.carray)
				carray_ext._dot_mat_carray(self, matrix, matrix[0], out)

			return out

	def empty_like_dot(self, matrix, chunklen=None, cparams=None, rootdir=None):
		'''
		Create en empty bvec.carray for use with the out parameter
		of the dot method. Allows saving to disk, selection of
		compression ratio and modification of other carray parameters.

		This is a relatively cheap operation.
		'''
		if chunklen == None:
			chunklen = self.chunklen

		if type(matrix) == np.ndarray:

			assert len(matrix.shape) == 1

			return self.empty_like(shape=(self.shape[0],), chunklen=chunklen, cparams=cparams, rootdir=rootdir)

		elif isinstance(matrix, bcolz.carray):

			assert len(matrix.shape) == 2

			return self.empty_like(shape=(self.shape[0], matrix.shape[0]), chunklen=chunklen, cparams=cparams, rootdir=rootdir)



	def empty_like(self, shape=None, chunklen=None, cparams=None, rootdir=None):
		'''
		Create an empty bvec.carray container matching this one, with optional
		modifications.
		'''

		p_dtype = self.dtype
		if shape == None:
			shape = self.shape
		if cparams == None:
			cparams = self.cparams

		if(len(shape) == 1):

			result_template = np.ndarray(shape=(0), dtype=p_dtype)
			return bvec.carray(result_template, expectedlen=shape[0], chunklen=chunklen, cparams=cparams, rootdir=rootdir)


		elif(len(self.shape) == 2):

			result_template = np.ndarray((0, shape[1]), dtype=p_dtype)
			return bvec.carray(result_template, expectedlen=shape[0], chunklen=chunklen, cparams=cparams, rootdir=rootdir)


		else:
			raise ValueError("Can't create a carray like that. Only one and two dimensions supported.")
