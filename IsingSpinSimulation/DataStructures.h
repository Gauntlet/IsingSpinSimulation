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

	enum class MemoryLocation { host, device };

	template<class elem_type> class Matrix
	{
		private:
		const MemoryLocation *_memloc;
		elem_type* _data;
		uint32_t* _length;

		uint32_t* _numOfCols;
		uint32_t* _numOfRows;

		void initialize( const uint32_t numofcols, const uint32_t numofrows, const MemoryLocation memloc );

		public:
		Matrix( const uint32_t N, const MemoryLocation memloc );
		Matrix( const uint32_t numofcols, const uint32_t numofrows, const MemoryLocation memloc );

		~Matrix();

		CUDA_CALLABLE_MEMBER MemoryLocation memLoc() const;
		CUDA_CALLABLE_MEMBER elem_type get( const uint32_t row, const uint32_t col ) const;
		CUDA_CALLABLE_MEMBER void set( const uint32_t row, const uint32_t col, const elem_type value );

		CUDA_CALLABLE_MEMBER uint32_t length() const;
		CUDA_CALLABLE_MEMBER uint32_t numOfColumns() const;
		CUDA_CALLABLE_MEMBER uint32_t numOfRows() const;
	};

	template <class elem_type> class MatrixShared
	{
		private:
		Matrix *intermediary;

		void initialize( const uint32_t numofcols, const uint32_t numofrows );
		public:
		Matrix *host;
		Matrix *device;

		MatrixShared( const uint32_t N );
		MatrixShared( const uint32_t numofcols, const uint32_t numofrows );
		~MatrixShared();

		void host2device();
		void device2host();

	};

	template <class elem_type> class JaggedList
	{
		private:
		MemoryLocation *_memloc;
		elem_type* _data;
		uint32_t* _length;

		uint32_t* _lengths;
		uint32_t* _offsets;

		public:
		JaggedList( const uint32_t N, const uint32_t* lengths, const MemoryLocation memloc );

		~JaggedList();

		CUDA_CALLABLE_MEMBER MemoryLocation memLoc() const;
		CUDA_CALLABLE_MEMBER elem_type get( const uint32_t row, const uint32_t col ) const;
		CUDA_CALLABLE_MEMBER void set( const uint32_t row, const uint32_t col, const elem_type val );

		CUDA_CALLABLE_MEMBER uint32_t length() const;
		CUDA_CALLABLE_MEMBER uint32_t size() const;
		CUDA_CALLABLE_MEMBER uint32_t length( const uint32_t row ) const;
		CUDA_CALLABLE_MEMBER uint32_t offset( const uint32_t row ) const;
	};

	template <class elem_type> class JaggedListShared
	{
		template <class elem_type2>
		friend class JaggedList;

		private:
		JaggedList *intermediary;

		public:
		JaggedList *host;
		JaggedList *device;

		JaggedListShared( const uint32_t N, const uint32_t* lengths );
		~JaggedListShared();

		void host2device();
		void device2host();

	};
}

#endif