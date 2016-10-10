#ifndef MATRIX_SHARED_H
#define MATRIX_SHARED_H

namespace kspace
{
	template <class elem_type> 
	class MatrixShared
	{
		friend class MatrixShared<elem_type>;
	private:
		Matrix<elem_type>* intermediary_ptr;
		Matrix<elem_type>* host_ptr;
		Matrix<elem_type>* device_ptr;
		
	protected:
		void initialise( const uint32_t number_of_columns, const uint32_t number_of_rows );
		void move_data( MatrixShared<elem_type>&& that );

	public:
		MatrixShared( const uint32_t N );
		MatrixShared( const uint32_t number_of_columns, const uint32_t number_of_rows );
		~MatrixShared();

		//delete copy constructors and assignment operator
		MatrixShared( const MatrixShared<elem_type>& ) = delete;
		MatrixShared<elem_type>& operator=( const MatrixShared<elem_type>& ) = delete;

		//define move constructor and move operator
		MatrixShared( MatrixShared<elem_type>&& that ) { move_data( that ); }
		MatrixShared<elem_type>& operator=( MatrixShared<elem_type>&& that ) { move_data( that ); }

		Matrix<elem_type>& host() { return *host_ptr; }
		Matrix<elem_type>& device() { return *device_ptr; }

		void host2device();
		void device2host();
	};
}

#endif