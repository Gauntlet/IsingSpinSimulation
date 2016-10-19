#ifndef MATRIX_SHARED_H
#define MATRIX_SHARED_H

#include "Matrix.h"

namespace kspace
{
	template <class elem_type> 
	class MatrixManager
	{
		friend class Matrix<elem_type>::SET<elem_type>;
	private:
		Matrix<elem_type>* intermediary_ptr;	/**< A pointer to a Matrix on the host which provides access to data stored in the device memory. */
		Matrix<elem_type>* host_ptr;			/**< A pointer to a Matrix on the host which provides access to the data stored in the host memory. */
		Matrix<elem_type>* device_ptr;			/**< A pointer to a Matrix on the device which provides access to data stored in the device memory */

	protected:
		/**
		* Creates a number_of_columns x number_of_rows Matrix on the host with the data stored on the host.
		* @param number_of_columns an integer.
		* @param number_of_rows an integer.
		*/
		void initialise_host( const uint32_t number_of_columns, const uint32_t number_of_rows );

		/**
		* Creates a number_of_columns x number_of_rows Matrix on the host with the data stored on the device.
		* @param number_of_columns an integer.
		* @param number_of_rows an integer.
		*/
		void initialise_intermediary( const uint32_t number_of_columns, const uint32_t number_of_rows );

		/**
		* Creates a Matrix on the device which handles the same pointers as the intermediary Matrix.
		*/
		void initialise_device();

		/**
		* Moves the pointers of the data stored in another MatrixManager object to this one.
		*/
		void move_data( MatrixManager<elem_type>&& that );

		/**
		* Moves the pointers of the data stored in Matrix object and creates the appropriate missing
		* Matrices.
		*/
		void move_data( Matrix<elem_type>&& that );

		/**
		* Frees the memory used by the matrices manged by this MatrixManager object.
		*/
		void clear();

		/**
		* Returns a reference to the intermediary Matrix object.
		*/
		Matrix<elem_type>& intermediary() { return *intermediary_ptr; }


	public:
		/**
		* Parameterised constructor.
		*
		* Creates a square NxN matrix on the host and device.
		* @param N a 4 byte integer.
		*/
		MatrixManager( const uint32_t N );

		/**
		* Parameterised constructor.
		*
		* Creates a rectangular number_of_columns x number_of_rows matrix on the host and device.
		* @param number_of_columns a 4 byte integer.
		* @param number_of_rows a 4 byte integer.
		*/
		MatrixManager( const uint32_t number_of_columns, const uint32_t number_of_rows );

		/**
		* Destructor.
		*
		* Frees the memory used by the Matrices managed by this MatrixManager object.
		*/
		~MatrixManager();

		/**
		* Delete copy constructor to prevent multiple MatrixManager objects from managing the same Matrices.
		*/
		MatrixManager( const MatrixManager<elem_type>& ) = delete;

		/**
		* Delete copy assignment operator to prevent multiple MatrixManager objects from managing the same Matrices.
		*/
		MatrixManager<elem_type>& operator=( const MatrixManager<elem_type>& ) = delete;

		/**
		* Move constructor.
		*
		* Moves the pointers to the Matrices managed by the passed MatrixManager to the one being constructed.
		* @param that a MatrixManager object.
		*/
		MatrixManager( MatrixManager<elem_type>&& that ) { move_data( that ); }

		/**
		* Move assignment operator.
		*
		* Moves the pointers to the Matrices managed by the MatrixManager on the RHS to the one on the LHS
		* @param that a MatrixManager object.
		*/
		MatrixManager<elem_type>& operator=( MatrixManager<elem_type>&& that ) { move_data( that ); }

		/**
		* Move constructor.
		*
		* Moves data from the Matrix object passed into ones intialised and managed by the MatrixManager object being constructed.
		* @param that a Matrix object.
		*/
		MatrixManager( Matrix<elem_type>&& that );

		/**
		* Move assignment operator.
		*
		* Moves data from the Matrix object on the RHS into ones intialised and managed by the MatrixManager object on the LHS.
		* @param that a Matrix object.
		*/
		MatrixManager<elem_type>& operator=( Matrix<elem_type>&& that );

		/**
		* Access to the Matrix data stored on the host.
		* @return reference to a Matrix.
		*/
		Matrix<elem_type>& host() { return *host_ptr; }

		/**
		* Access to the Matrix data stored on the device.
		* @return reference to a Matrix.
		*/
		Matrix<elem_type>& device() { return *device_ptr; }

		/**
		* Copies the values of the elements of the matrix to the device from the host.
		*/
		void host2device();

		/**
		* Copies the values of the elements of the matrix to the host from the device.
		*/
		void device2host();

	};
}

#endif