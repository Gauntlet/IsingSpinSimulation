#ifndef MATRIX_SHARED_H
#define MATRIX_SHARED_H

#include "Matrix.h"

namespace kspace
{
	template <class elem_type> 
	class MatrixManager
	{
		friend class Matrix<elem_type>;
	private:
		Matrix<elem_type>* intermediary_ptr;	/**< A pointer to a Matrix on the host which provides access to data stored in the device memory. */
		Matrix<elem_type>* host_ptr;			/**< A pointer to a Matrix on the host which provides access to the data stored in the host memory. */
		Matrix<elem_type>* device_ptr;			/**< A pointer to a Matrix on the device which provides access to data stored in the device memory */

	private:
		
		void initialise_host( const uint32_t number_of_columns, const uint32_t number_of_rows );
		void initialise_intermediary( const uint32_t number_of_columns, const uint32_t number_of_rows );
		void initialise_device();
		void move_data( MatrixManager<elem_type>&& that );
		void move_data( Matrix<elem_type>&& that );
		void clear();
		Matrix<elem_type>& intermediary();

	public:
		
		MatrixManager( const uint32_t N );
		MatrixManager( const uint32_t number_of_columns, const uint32_t number_of_rows );

		~MatrixManager();

		/**
		* Delete copy constructor to prevent multiple MatrixManager objects from managing the same Matrices.
		*/
		MatrixManager( const MatrixManager<elem_type>& ) = delete;

		/**
		* Delete copy assignment operator to prevent multiple MatrixManager objects from managing the same Matrices.
		*/
		MatrixManager<elem_type>& operator=( const MatrixManager<elem_type>& ) = delete;

		MatrixManager( MatrixManager<elem_type>&& that );
		MatrixManager( Matrix<elem_type>&& that );
		
		MatrixManager<elem_type>& operator=( MatrixManager<elem_type>&& that );
		MatrixManager<elem_type>& operator=( Matrix<elem_type>&& that );

		Matrix<elem_type>& host();
		Matrix<elem_type>& device();

		
		void host2device();
		void device2host();

	};
}

#endif