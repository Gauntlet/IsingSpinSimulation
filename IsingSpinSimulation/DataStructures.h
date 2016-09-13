#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstdint>

#ifdef __CUDA_ARCH__
	#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
	#define CUDA_CALLABLE_MEMBER
#endif 

static void HandleError( cudaError_t err, const char *file, int line )
{
	if ( err != cudaSuccess ) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

namespace kspace
{

	enum class MemoryLocation : std::uint32_t { host, device };

	template<class elem_type> class Matrix
	{
		friend class MatrixShared < elem_type > ;

	private:
		const MemoryLocation *_memloc;
		elem_type* _data;
		uint32_t* _length;

		uint32_t* _number_of_columns;
		uint32_t* _number_of_rows;

		void initialise(const uint32_t num_of_rows, const uint32_t num_of_columns, const MemoryLocation memloc);

	public:
		Matrix( const uint32_t N, const MemoryLocation memloc );
		Matrix(const uint32_t num_of_rows, const uint32_t num_of_columns, const MemoryLocation memloc);

		~Matrix();

		CUDA_CALLABLE_MEMBER MemoryLocation memory_location() const;
		CUDA_CALLABLE_MEMBER elem_type get( const uint32_t row, const uint32_t column ) const;
		CUDA_CALLABLE_MEMBER void set( const uint32_t row, const uint32_t column, const elem_type value );

		CUDA_CALLABLE_MEMBER uint32_t length() const;
		CUDA_CALLABLE_MEMBER uint32_t number_of_columns() const;
		CUDA_CALLABLE_MEMBER uint32_t number_of_rows() const;
	};

	template <class elem_type> class MatrixShared
	{
	private:
		Matrix<elem_type> *intermediary;
		void initialise(const uint32_t num_of_rows, const uint32_t num_of_columns);
	public:
		Matrix<elem_type> *host;
		Matrix<elem_type> *device;

		MatrixShared( const uint32_t N );
		MatrixShared(const uint32_t num_of_rows, const uint32_t num_of_columns);
		~MatrixShared();

		void host2device();
		void device2host();
	};

	template <class elem_type> class JaggedList
	{
		friend class JaggedListShared < elem_type > ;

		private:
		MemoryLocation *_memloc;
		elem_type* _data;
		uint32_t* _length;

		uint32_t* _lengths;
		uint32_t* _offsets;

		public:
		JaggedList( const uint32_t N, const uint32_t* lengths, const MemoryLocation memloc );

		~JaggedList();

		CUDA_CALLABLE_MEMBER MemoryLocation memory_location() const;
		CUDA_CALLABLE_MEMBER elem_type get( const uint32_t row, const uint32_t column ) const;
		CUDA_CALLABLE_MEMBER void set( const uint32_t row, const uint32_t column, const elem_type val );

		CUDA_CALLABLE_MEMBER uint32_t length() const;
		CUDA_CALLABLE_MEMBER uint32_t size() const;
		CUDA_CALLABLE_MEMBER uint32_t length( const uint32_t row ) const;
		CUDA_CALLABLE_MEMBER uint32_t offset( const uint32_t row ) const;
	};

	template <class elem_type> class JaggedListShared
	{
		private:
		JaggedList<elem_type> *intermediary;

		public:
		JaggedList<elem_type> *host;
		JaggedList<elem_type> *device;

		JaggedListShared( const uint32_t N, const uint32_t* lengths );
		~JaggedListShared();

		void host2device();
		void device2host();

	};
}

#endif